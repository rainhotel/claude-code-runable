import type Anthropic from '@anthropic-ai/sdk'
import type { BetaToolUnion } from '@anthropic-ai/sdk/resources/beta/messages.js'
import { randomUUID } from 'crypto'
import {
  getLastApiCompletionTimestamp,
  setLastApiCompletionTimestamp,
} from '../bootstrap/state.js'
import { STRUCTURED_OUTPUTS_BETA_HEADER } from '../constants/betas.js'
import type { QuerySource } from '../constants/querySource.js'
import {
  getAttributionHeader,
  getCLISyspromptPrefix,
} from '../constants/system.js'
import { logEvent } from '../services/analytics/index.js'
import type { AnalyticsMetadata_I_VERIFIED_THIS_IS_NOT_CODE_OR_FILEPATHS } from '../services/analytics/metadata.js'
import { getAPIMetadata } from '../services/api/claude.js'
import { getAnthropicClient } from '../services/api/client.js'
import { getGrokClient } from '../services/api/grok/client.js'
import { resolveGrokModel } from '../services/api/grok/modelMapping.js'
import { getOpenAIClient } from '../services/api/openai/client.js'
import { anthropicMessagesToOpenAI } from '../services/api/openai/convertMessages.js'
import {
  anthropicToolChoiceToOpenAI,
  anthropicToolsToOpenAI,
} from '../services/api/openai/convertTools.js'
import { resolveOpenAIModel } from '../services/api/openai/modelMapping.js'
import { getModelBetas, modelSupportsStructuredOutputs } from './betas.js'
import { computeFingerprint } from './fingerprint.js'
import { safeParseJSON } from './json.js'
import { createAssistantMessage, createUserMessage } from './messages.js'
import { normalizeModelStringForAPI } from './model/model.js'
import { getAPIProvider } from './model/providers.js'
import { asSystemPrompt } from './systemPromptType.js'

type MessageParam = Anthropic.MessageParam
type TextBlockParam = Anthropic.TextBlockParam
type Tool = Anthropic.Tool
type ToolChoice = Anthropic.ToolChoice
type BetaMessage = Anthropic.Beta.Messages.BetaMessage
type BetaJSONOutputFormat = Anthropic.Beta.Messages.BetaJSONOutputFormat
type BetaThinkingConfigParam = Anthropic.Beta.Messages.BetaThinkingConfigParam

export type SideQueryOptions = {
  /** Model to use for the query */
  model: string
  /**
   * System prompt - string or array of text blocks (will be prefixed with CLI attribution).
   *
   * The attribution header is always placed in its own TextBlockParam block to ensure
   * server-side parsing correctly extracts the cc_entrypoint value without including
   * system prompt content.
   */
  system?: string | TextBlockParam[]
  /** Messages to send (supports cache_control on content blocks) */
  messages: MessageParam[]
  /** Optional tools (supports both standard Tool[] and BetaToolUnion[] for custom tool types) */
  tools?: Tool[] | BetaToolUnion[]
  /** Optional tool choice (use { type: 'tool', name: 'x' } for forced output) */
  tool_choice?: ToolChoice
  /** Optional JSON output format for structured responses */
  output_format?: BetaJSONOutputFormat
  /** Max tokens (default: 1024) */
  max_tokens?: number
  /** Max retries (default: 2) */
  maxRetries?: number
  /** Abort signal */
  signal?: AbortSignal
  /** Skip CLI system prompt prefix (keeps attribution header for OAuth). For internal classifiers that provide their own prompt. */
  skipSystemPromptPrefix?: boolean
  /** Temperature override */
  temperature?: number
  /** Thinking budget (enables thinking), or `false` to send `{ type: 'disabled' }`. */
  thinking?: number | false
  /** Stop sequences — generation stops when any of these strings is emitted */
  stop_sequences?: string[]
  /** Attributes this call in tengu_api_success for COGS joining against reporting.sampling_calls. */
  querySource: QuerySource
}

/**
 * Extract text from first user message for fingerprint computation.
 */
function extractFirstUserMessageText(messages: MessageParam[]): string {
  const firstUserMessage = messages.find(m => m.role === 'user')
  if (!firstUserMessage) return ''

  const content = firstUserMessage.content
  if (typeof content === 'string') return content

  // Array of content blocks - find first text block
  const textBlock = content.find(block => block.type === 'text')
  return textBlock?.type === 'text' ? textBlock.text : ''
}

function toSystemPrompt(system?: string | TextBlockParam[]) {
  return asSystemPrompt(
    (Array.isArray(system)
      ? system.map(block => block.text)
      : system
        ? [system]
        : []
    ).filter(Boolean),
  )
}

function toInternalMessages(messages: MessageParam[]) {
  return messages.flatMap(message => {
    if (message.role === 'assistant') {
      return [
        createAssistantMessage({
          content:
            typeof message.content === 'string'
              ? message.content
              : (message.content as BetaMessage['content']),
        }),
      ]
    }

    if (message.role === 'user') {
      return [
        createUserMessage({
          content:
            typeof message.content === 'string'
              ? message.content
              : message.content,
        }),
      ]
    }

    return []
  })
}

function extractOpenAIResponseText(content: unknown): string {
  if (typeof content === 'string') {
    return content
  }

  if (!Array.isArray(content)) {
    return ''
  }

  return content
    .map(part => {
      if (!part || typeof part !== 'object') {
        return ''
      }
      if ('text' in part && typeof part.text === 'string') {
        return part.text
      }
      if ('refusal' in part && typeof part.refusal === 'string') {
        return part.refusal
      }
      return ''
    })
    .filter(Boolean)
    .join('\n')
}

function normalizeOpenAIToolInput(input: string): string | unknown {
  const parsed = safeParseJSON(input)
  return parsed === null && input.length > 0 ? input : (parsed ?? {})
}

function mapOpenAIFinishReason(
  reason: string | null | undefined,
  hasToolCalls: boolean,
): BetaMessage['stop_reason'] {
  if (hasToolCalls) {
    return 'tool_use'
  }

  switch (reason) {
    case 'length':
      return 'max_tokens'
    case 'tool_calls':
      return 'tool_use'
    case 'stop':
    case 'content_filter':
    default:
      return 'end_turn'
  }
}

async function sideQueryViaOpenAICompatibleProvider(
  opts: SideQueryOptions,
  provider: 'grok' | 'openai',
): Promise<BetaMessage> {
  const internalMessages = toInternalMessages(opts.messages)
  const systemPrompt = toSystemPrompt(opts.system)
  const openaiMessages = anthropicMessagesToOpenAI(
    internalMessages,
    systemPrompt,
  )
  const openaiTools = opts.tools
    ? anthropicToolsToOpenAI(opts.tools as BetaToolUnion[])
    : undefined
  const openaiToolChoice = anthropicToolChoiceToOpenAI(opts.tool_choice)
  const resolvedModel =
    provider === 'grok'
      ? resolveGrokModel(opts.model)
      : resolveOpenAIModel(opts.model)

  const client =
    provider === 'grok'
      ? getGrokClient({
          maxRetries: opts.maxRetries ?? 2,
          source: opts.querySource,
        })
      : getOpenAIClient({
          maxRetries: opts.maxRetries ?? 2,
          source: opts.querySource,
        })

  const responseFormat = opts.output_format
    ? {
        type: 'json_schema' as const,
        json_schema: {
          name: 'side_query',
          schema: opts.output_format.schema,
          strict: true,
        },
      }
    : undefined

  const response = await client.chat.completions.create(
    {
      model: resolvedModel,
      messages: openaiMessages,
      ...(openaiTools && openaiTools.length > 0 && { tools: openaiTools }),
      ...(openaiToolChoice && { tool_choice: openaiToolChoice }),
      ...(responseFormat && { response_format: responseFormat }),
      ...(opts.max_tokens !== undefined && {
        max_completion_tokens: opts.max_tokens,
      }),
      ...(opts.temperature !== undefined && { temperature: opts.temperature }),
      ...(opts.stop_sequences && { stop: opts.stop_sequences }),
    },
    {
      signal: opts.signal,
    },
  )

  const choice = response.choices[0]
  const toolCalls = choice?.message.tool_calls ?? []
  const text = extractOpenAIResponseText(choice?.message.content)
  const content: BetaMessage['content'] = []

  if (text) {
    content.push({ type: 'text', text })
  }

  for (const toolCall of toolCalls) {
    if (toolCall.type !== 'function') {
      continue
    }

    content.push({
      type: 'tool_use',
      id: toolCall.id ?? randomUUID(),
      name: toolCall.function.name,
      input: normalizeOpenAIToolInput(toolCall.function.arguments),
    })
  }

  return {
    id: response.id ?? randomUUID(),
    type: 'message',
    role: 'assistant',
    model: response.model || resolvedModel,
    content,
    stop_reason: mapOpenAIFinishReason(
      choice?.finish_reason,
      toolCalls.length > 0,
    ),
    stop_sequence: null,
    usage: {
      input_tokens: response.usage?.prompt_tokens ?? 0,
      output_tokens: response.usage?.completion_tokens ?? 0,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
    },
  } as BetaMessage
}

/**
 * Lightweight API wrapper for "side queries" outside the main conversation loop.
 *
 * Use this instead of direct client.beta.messages.create() calls to ensure
 * proper OAuth token validation with fingerprint attribution headers.
 *
 * This handles:
 * - Fingerprint computation for OAuth validation
 * - Attribution header injection
 * - CLI system prompt prefix
 * - Proper betas for the model
 * - API metadata
 * - Model string normalization (strips [1m] suffix for API)
 *
 * @example
 * // Permission explainer
 * await sideQuery({ querySource: 'permission_explainer', model, system: SYSTEM_PROMPT, messages, tools, tool_choice })
 *
 * @example
 * // Session search
 * await sideQuery({ querySource: 'session_search', model, system: SEARCH_PROMPT, messages })
 *
 * @example
 * // Model validation
 * await sideQuery({ querySource: 'model_validation', model, max_tokens: 1, messages: [{ role: 'user', content: 'Hi' }] })
 */
export async function sideQuery(opts: SideQueryOptions): Promise<BetaMessage> {
  const {
    model,
    system,
    messages,
    tools,
    tool_choice,
    output_format,
    max_tokens = 1024,
    maxRetries = 2,
    signal,
    skipSystemPromptPrefix,
    temperature,
    thinking,
    stop_sequences,
  } = opts

  const provider = getAPIProvider()
  const normalizedModel = normalizeModelStringForAPI(model)
  const start = Date.now()
  const response =
    provider === 'openai' || provider === 'grok'
      ? await sideQueryViaOpenAICompatibleProvider(opts, provider)
      : await (async () => {
          const client = await getAnthropicClient({
            maxRetries,
            model,
            source: 'side_query',
          })
          const betas = [...getModelBetas(model)]
          if (
            output_format &&
            modelSupportsStructuredOutputs(model) &&
            !betas.includes(STRUCTURED_OUTPUTS_BETA_HEADER)
          ) {
            betas.push(STRUCTURED_OUTPUTS_BETA_HEADER)
          }

          const messageText = extractFirstUserMessageText(messages)
          const fingerprint = computeFingerprint(messageText, MACRO.VERSION)
          const attributionHeader = getAttributionHeader(fingerprint)

          const systemBlocks: TextBlockParam[] = [
            attributionHeader ? { type: 'text', text: attributionHeader } : null,
            ...(skipSystemPromptPrefix
              ? []
              : [
                  {
                    type: 'text' as const,
                    text: getCLISyspromptPrefix({
                      isNonInteractive: false,
                      hasAppendSystemPrompt: false,
                    }),
                  },
                ]),
            ...(Array.isArray(system)
              ? system
              : system
                ? [{ type: 'text' as const, text: system }]
                : []),
          ].filter((block): block is TextBlockParam => block !== null)

          let thinkingConfig: BetaThinkingConfigParam | undefined
          if (thinking === false) {
            thinkingConfig = { type: 'disabled' }
          } else if (thinking !== undefined) {
            thinkingConfig = {
              type: 'enabled',
              budget_tokens: Math.min(thinking, max_tokens - 1),
            }
          }

          // biome-ignore lint/plugin: this IS the wrapper that handles OAuth attribution
          return client.beta.messages.create(
            {
              model: normalizedModel,
              max_tokens,
              system: systemBlocks,
              messages,
              ...(tools && { tools }),
              ...(tool_choice && { tool_choice }),
              ...(output_format && { output_config: { format: output_format } }),
              ...(temperature !== undefined && { temperature }),
              ...(stop_sequences && { stop_sequences }),
              ...(thinkingConfig && { thinking: thinkingConfig }),
              ...(betas.length > 0 && { betas }),
              metadata: getAPIMetadata(),
            },
            { signal },
          )
        })()

  const requestId =
    (response as { _request_id?: string | null })._request_id ?? undefined
  const now = Date.now()
  const lastCompletion = getLastApiCompletionTimestamp()
  logEvent('tengu_api_success', {
    requestId:
      requestId as AnalyticsMetadata_I_VERIFIED_THIS_IS_NOT_CODE_OR_FILEPATHS,
    querySource:
      opts.querySource as AnalyticsMetadata_I_VERIFIED_THIS_IS_NOT_CODE_OR_FILEPATHS,
    model:
      normalizedModel as AnalyticsMetadata_I_VERIFIED_THIS_IS_NOT_CODE_OR_FILEPATHS,
    inputTokens: response.usage.input_tokens,
    outputTokens: response.usage.output_tokens,
    cachedInputTokens: response.usage.cache_read_input_tokens ?? 0,
    uncachedInputTokens: response.usage.cache_creation_input_tokens ?? 0,
    durationMsIncludingRetries: now - start,
    timeSinceLastApiCallMs:
      lastCompletion !== null ? now - lastCompletion : undefined,
  })
  setLastApiCompletionTimestamp(now)

  return response
}
