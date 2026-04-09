import { describe, expect, mock, test } from 'bun:test'
import { applyOpenAIMessageDelta, createOpenAIStream } from '../index.js'

function makeEmptyStream() {
  return {
    [Symbol.asyncIterator]() {
      return {
        async next() {
          return { done: true, value: undefined }
        },
      }
    },
  }
}

function makeAssistantMessage() {
  return {
    type: 'assistant' as const,
    requestId: undefined,
    uuid: 'assistant-uuid',
    timestamp: '2026-04-09T00:00:00.000Z',
    message: {
      id: 'msg_test',
      type: 'message' as const,
      role: 'assistant' as const,
      content: [{ type: 'text' as const, text: 'partial answer' }],
      model: 'deepseek-chat',
      stop_reason: null,
      stop_sequence: null,
      usage: {
        input_tokens: 5,
        output_tokens: 0,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
      },
    },
  }
}

describe('createOpenAIStream', () => {
  test('sends max_tokens for compatible OpenAI-style backends', async () => {
    const create = mock(async (_params: unknown) => makeEmptyStream())

    await createOpenAIStream(
      {
        chat: {
          completions: {
            create,
          },
        },
      } as any,
      {
        model: 'deepseek-chat',
        messages: [],
        stream: true,
        stream_options: { include_usage: true },
      } as any,
      777,
      new AbortController().signal,
      'deepseek-chat',
    )

    expect(create).toHaveBeenCalledTimes(1)
    const [requestParams] = create.mock.calls[0]!
    expect(requestParams.max_tokens).toBe(777)
    expect(requestParams.max_completion_tokens).toBeUndefined()
  })

  test('retries with max_tokens when max_completion_tokens is rejected', async () => {
    const create = mock(async (params: Record<string, unknown>) => {
      if ('max_completion_tokens' in params) {
        throw new Error('Unknown parameter: max_completion_tokens')
      }
      return makeEmptyStream()
    })

    await createOpenAIStream(
      {
        chat: {
          completions: {
            create,
          },
        },
      } as any,
      {
        model: 'gpt-5-mini',
        messages: [],
        stream: true,
        stream_options: { include_usage: true },
      } as any,
      2048,
      new AbortController().signal,
      'gpt-5-mini',
    )

    expect(create).toHaveBeenCalledTimes(2)
    const [firstRequest] = create.mock.calls[0]!
    const [secondRequest] = create.mock.calls[1]!
    expect(firstRequest.max_completion_tokens).toBe(2048)
    expect(secondRequest.max_tokens).toBe(2048)
  })
})

describe('applyOpenAIMessageDelta', () => {
  test('writes final stop_reason back and emits max_output_tokens api error', () => {
    const assistantMessage = makeAssistantMessage()

    const result = applyOpenAIMessageDelta({
      event: {
        delta: {
          stop_reason: 'max_tokens',
          stop_sequence: null,
        },
        usage: {
          output_tokens: 42,
        },
      },
      usage: {
        input_tokens: 5,
        output_tokens: 0,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
      },
      newMessages: [assistantMessage as any],
      maxOutputTokens: 2048,
      emittedMaxTokensError: false,
    })

    expect(assistantMessage.message.stop_reason).toBe('max_tokens')
    expect(assistantMessage.message.usage.output_tokens).toBe(42)
    expect(result.emittedMaxTokensError).toBe(true)
    expect(result.syntheticError?.apiError).toBe('max_output_tokens')
    expect(
      (result.syntheticError?.message.content[0] as { text: string }).text,
    ).toContain('2048 output token maximum')
  })
})
