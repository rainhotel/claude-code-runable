import type { BetaToolUnion } from '@anthropic-ai/sdk/resources/beta/messages/messages.mjs'
import type { ChatCompletionCreateParamsStreaming } from 'openai/resources/chat/completions/completions.mjs'
import type { SystemPrompt } from '../../../utils/systemPromptType.js'
import type {
  Message,
  StreamEvent,
  SystemAPIErrorMessage,
  AssistantMessage,
} from '../../../types/message.js'
import type { Tools } from '../../../Tool.js'
import { getOpenAIClient } from './client.js'
import { anthropicMessagesToOpenAI } from './convertMessages.js'
import {
  anthropicToolsToOpenAI,
  anthropicToolChoiceToOpenAI,
} from './convertTools.js'
import { adaptOpenAIStreamToAnthropic } from './streamAdapter.js'
import { resolveOpenAIModel } from './modelMapping.js'
import {
  createAssistantAPIErrorMessage,
  normalizeContentFromAPI,
  normalizeMessagesForAPI,
} from '../../../utils/messages.js'
import { toolToAPISchema } from '../../../utils/api.js'
import { getEmptyToolPermissionContext } from '../../../Tool.js'
import { logForDebugging } from '../../../utils/debug.js'
import { addToTotalSessionCost } from '../../../cost-tracker.js'
import { calculateUSDCost } from '../../../utils/modelCost.js'
import { getModelMaxOutputTokens } from '../../../utils/context.js'
import { validateBoundedIntEnvVar } from '../../../utils/envValidation.js'
import type { Options } from '../claude.js'
import { randomUUID } from 'crypto'
import { API_ERROR_MESSAGE_PREFIX } from '../errors.js'

type OpenAIMaxTokensParam = 'max_completion_tokens' | 'max_tokens'
type OpenAIUsage = {
  input_tokens: number
  output_tokens: number
  cache_creation_input_tokens: number
  cache_read_input_tokens: number
}

export function getOpenAIMaxOutputTokens(options: Options): number {
  if (options.maxOutputTokensOverride !== undefined) {
    return options.maxOutputTokensOverride
  }

  const maxOutputTokens = getModelMaxOutputTokens(options.model)
  return validateBoundedIntEnvVar(
    'CLAUDE_CODE_MAX_OUTPUT_TOKENS',
    process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS,
    maxOutputTokens.default,
    maxOutputTokens.upperLimit,
  ).effective
}

export function prefersMaxCompletionTokens(model: string): boolean {
  return /^(o\d|gpt-5(?:-|$))/i.test(model)
}

export function shouldRetryWithAlternateMaxTokensParam(
  error: unknown,
  attemptedParam: OpenAIMaxTokensParam,
): boolean {
  const message = (
    error instanceof Error ? error.message : String(error)
  ).toLowerCase()

  if (attemptedParam === 'max_tokens') {
    return (
      message.includes('max_tokens') &&
      (message.includes('max_completion_tokens') ||
        message.includes('o-series') ||
        message.includes('reasoning model') ||
        message.includes('incompatible'))
    )
  }

  return (
    message.includes('max_completion_tokens') &&
    (message.includes('unsupported') ||
      message.includes('unknown') ||
      message.includes('unrecognized') ||
      message.includes('extra inputs') ||
      message.includes('not allowed') ||
      message.includes('not permitted'))
  )
}

export async function createOpenAIStream(
  client: ReturnType<typeof getOpenAIClient>,
  requestBase: Omit<
    ChatCompletionCreateParamsStreaming,
    'max_completion_tokens' | 'max_tokens'
  >,
  maxOutputTokens: number,
  signal: AbortSignal,
  model: string,
) {
  let tokenLimitParam: OpenAIMaxTokensParam = prefersMaxCompletionTokens(model)
    ? 'max_completion_tokens'
    : 'max_tokens'

  for (let attempt = 0; attempt < 2; attempt++) {
    const requestParams = {
      ...requestBase,
      [tokenLimitParam]: maxOutputTokens,
    } as ChatCompletionCreateParamsStreaming

    try {
      return await client.chat.completions.create(requestParams, {
        signal,
      })
    } catch (error) {
      if (
        attempt === 0 &&
        shouldRetryWithAlternateMaxTokensParam(error, tokenLimitParam)
      ) {
        const retryParam: OpenAIMaxTokensParam =
          tokenLimitParam === 'max_tokens'
            ? 'max_completion_tokens'
            : 'max_tokens'
        logForDebugging(
          `[OpenAI] Retrying model=${model} with ${retryParam} after ${tokenLimitParam} was rejected`,
        )
        tokenLimitParam = retryParam
        continue
      }
      throw error
    }
  }

  throw new Error('OpenAI stream creation exhausted retries unexpectedly')
}

export function applyOpenAIMessageDelta(params: {
  event: {
    delta?: { stop_reason?: string | null }
    usage?: Partial<OpenAIUsage>
  }
  usage: OpenAIUsage
  newMessages: AssistantMessage[]
  maxOutputTokens: number
  emittedMaxTokensError: boolean
}): {
  usage: OpenAIUsage
  emittedMaxTokensError: boolean
  syntheticError?: SystemAPIErrorMessage
} {
  const nextUsage = params.event.usage
    ? { ...params.usage, ...params.event.usage }
    : params.usage
  const stopReason = params.event.delta?.stop_reason
  const lastMsg = params.newMessages.at(-1)

  if (lastMsg) {
    lastMsg.message.usage = nextUsage as typeof lastMsg.message.usage
    if (stopReason !== undefined) {
      lastMsg.message.stop_reason = stopReason
    }
  }

  if (stopReason === 'max_tokens' && !params.emittedMaxTokensError) {
    return {
      usage: nextUsage,
      emittedMaxTokensError: true,
      syntheticError: createAssistantAPIErrorMessage({
        content:
          `${API_ERROR_MESSAGE_PREFIX}: The model response exceeded the ` +
          `${params.maxOutputTokens} output token maximum. To configure this behavior, ` +
          `set the CLAUDE_CODE_MAX_OUTPUT_TOKENS environment variable.`,
        apiError: 'max_output_tokens',
        error: 'max_output_tokens',
      }),
    }
  }

  return {
    usage: nextUsage,
    emittedMaxTokensError: params.emittedMaxTokensError,
  }
}

/**
 * OpenAI-compatible query path. Converts Anthropic-format messages/tools to
 * OpenAI format, calls the OpenAI-compatible endpoint, and converts the
 * SSE stream back to Anthropic BetaRawMessageStreamEvent for consumption
 * by the existing query pipeline.
 */
export async function* queryModelOpenAI(
  messages: Message[],
  systemPrompt: SystemPrompt,
  tools: Tools,
  signal: AbortSignal,
  options: Options,
): AsyncGenerator<
  StreamEvent | AssistantMessage | SystemAPIErrorMessage,
  void
> {
  try {
    // 1. Resolve model name
    const openaiModel = resolveOpenAIModel(options.model)

    // 2. Normalize messages using shared preprocessing
    const messagesForAPI = normalizeMessagesForAPI(messages, tools)

    // 3. Build tool schemas
    const toolSchemas = await Promise.all(
      tools.map(tool =>
        toolToAPISchema(tool, {
          getToolPermissionContext: options.getToolPermissionContext,
          tools,
          agents: options.agents,
          allowedAgentTypes: options.allowedAgentTypes,
          model: options.model,
        }),
      ),
    )
    // Filter out non-standard tools (server tools like advisor)
    const standardTools = toolSchemas.filter(
      (t): t is BetaToolUnion & { type: string } => {
        const anyT = t as Record<string, unknown>
        return (
          anyT.type !== 'advisor_20260301' && anyT.type !== 'computer_20250124'
        )
      },
    )

    // 4. Convert messages and tools to OpenAI format
    const openaiMessages = anthropicMessagesToOpenAI(
      messagesForAPI,
      systemPrompt,
    )
    const openaiTools = anthropicToolsToOpenAI(standardTools)
    const openaiToolChoice = anthropicToolChoiceToOpenAI(options.toolChoice)
    const maxOutputTokens = getOpenAIMaxOutputTokens(options)

    // 5. Get client and make streaming request
    const client = getOpenAIClient({
      maxRetries: 0,
      fetchOverride: options.fetchOverride,
      source: options.querySource,
    })

    logForDebugging(
      `[OpenAI] Calling model=${openaiModel}, messages=${openaiMessages.length}, tools=${openaiTools.length}`,
    )

    // 6. Call OpenAI API with streaming
    const requestBase = {
      model: openaiModel,
      messages: openaiMessages,
      ...(openaiTools.length > 0 && {
        tools: openaiTools,
        ...(openaiToolChoice && { tool_choice: openaiToolChoice }),
      }),
      stream: true,
      stream_options: { include_usage: true },
      ...(options.temperatureOverride !== undefined && {
        temperature: options.temperatureOverride,
      }),
    } as Omit<
      ChatCompletionCreateParamsStreaming,
      'max_completion_tokens' | 'max_tokens'
    >

    const stream = await createOpenAIStream(
      client,
      requestBase,
      maxOutputTokens,
      signal,
      openaiModel,
    )

    // 7. Convert OpenAI stream to Anthropic events, then process into
    //    AssistantMessage + StreamEvent (matching the Anthropic path behavior)
    const adaptedStream = adaptOpenAIStreamToAnthropic(stream, openaiModel)

    // Accumulate content blocks and usage, same as the Anthropic path in claude.ts
    const contentBlocks: Record<number, any> = {}
    const newMessages: AssistantMessage[] = []
    let partialMessage: any
    let usage: OpenAIUsage = {
      input_tokens: 0,
      output_tokens: 0,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
    }
    let emittedMaxTokensError = false
    let ttftMs = 0
    const start = Date.now()

    for await (const event of adaptedStream) {
      switch (event.type) {
        case 'message_start': {
          partialMessage = (event as any).message
          ttftMs = Date.now() - start
          if ((event as any).message?.usage) {
            usage = {
              ...usage,
              ...(event as any).message.usage,
            }
          }
          break
        }
        case 'content_block_start': {
          const idx = (event as any).index
          const cb = (event as any).content_block
          if (cb.type === 'tool_use') {
            contentBlocks[idx] = { ...cb, input: '' }
          } else if (cb.type === 'text') {
            contentBlocks[idx] = { ...cb, text: '' }
          } else if (cb.type === 'thinking') {
            contentBlocks[idx] = { ...cb, thinking: '', signature: '' }
          } else {
            contentBlocks[idx] = { ...cb }
          }
          break
        }
        case 'content_block_delta': {
          const idx = (event as any).index
          const delta = (event as any).delta
          const block = contentBlocks[idx]
          if (!block) break
          if (delta.type === 'text_delta') {
            block.text = (block.text || '') + delta.text
          } else if (delta.type === 'input_json_delta') {
            block.input = (block.input || '') + delta.partial_json
          } else if (delta.type === 'thinking_delta') {
            block.thinking = (block.thinking || '') + delta.thinking
          } else if (delta.type === 'signature_delta') {
            block.signature = delta.signature
          }
          break
        }
        case 'content_block_stop': {
          const idx = (event as any).index
          const block = contentBlocks[idx]
          if (!block || !partialMessage) break

          const m: AssistantMessage = {
            message: {
              ...partialMessage,
              content: normalizeContentFromAPI([block], tools, options.agentId),
            },
            requestId: undefined,
            type: 'assistant',
            uuid: randomUUID(),
            timestamp: new Date().toISOString(),
          }
          newMessages.push(m)
          yield m
          break
        }
        case 'message_delta': {
          const deltaResult = applyOpenAIMessageDelta({
            event: event as {
              delta?: { stop_reason?: string | null }
              usage?: Partial<OpenAIUsage>
            },
            usage,
            newMessages,
            maxOutputTokens,
            emittedMaxTokensError,
          })
          usage = deltaResult.usage
          emittedMaxTokensError = deltaResult.emittedMaxTokensError
          if (deltaResult.syntheticError) {
            yield deltaResult.syntheticError
          }
          break
        }
        case 'message_stop':
          break
      }

      // Track cost and token usage (matching the Anthropic path in claude.ts)
      if (
        event.type === 'message_stop' &&
        usage.input_tokens + usage.output_tokens > 0
      ) {
        const costUSD = calculateUSDCost(openaiModel, usage as any)
        addToTotalSessionCost(costUSD, usage as any, options.model)
      }

      // Also yield as StreamEvent for real-time display (matching Anthropic path)
      yield {
        type: 'stream_event',
        event,
        ...(event.type === 'message_start' ? { ttftMs } : undefined),
      } as StreamEvent
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error)
    logForDebugging(`[OpenAI] Error: ${errorMessage}`, { level: 'error' })
    yield createAssistantAPIErrorMessage({
      content: `API Error: ${errorMessage}`,
      apiError: 'api_error',
      error: error instanceof Error ? error : new Error(String(error)),
    })
  }
}
