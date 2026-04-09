import { afterEach, beforeEach, describe, expect, mock, test } from 'bun:test'

const openAICreate = mock(async () => ({
  id: 'resp_123',
  model: 'gpt-5.4',
  choices: [
    {
      finish_reason: 'tool_calls',
      message: {
        content: 'Structured explanation ready',
        tool_calls: [
          {
            id: 'call_123',
            type: 'function',
            function: {
              name: 'explain_command',
              arguments:
                '{"riskLevel":"LOW","explanation":"safe","reasoning":"needed","risk":"none"}',
            },
          },
        ],
      },
    },
  ],
  usage: {
    prompt_tokens: 11,
    completion_tokens: 7,
  },
}))
const getOpenAIClient = mock(() => ({
  chat: {
    completions: {
      create: openAICreate,
    },
  },
}))

mock.module('../../services/api/openai/client.js', () => ({
  getOpenAIClient,
}))

const { sideQuery } = await import('../sideQuery.js')

describe('sideQuery', () => {
  const originalOpenAIFlag = process.env.CLAUDE_CODE_USE_OPENAI
  const originalOpenAIModel = process.env.OPENAI_MODEL

  beforeEach(() => {
    process.env.CLAUDE_CODE_USE_OPENAI = '1'
    process.env.OPENAI_MODEL = 'gpt-5.4'
    openAICreate.mockClear()
    getOpenAIClient.mockClear()
  })

  afterEach(() => {
    if (originalOpenAIFlag !== undefined) {
      process.env.CLAUDE_CODE_USE_OPENAI = originalOpenAIFlag
    } else {
      delete process.env.CLAUDE_CODE_USE_OPENAI
    }
    if (originalOpenAIModel !== undefined) {
      process.env.OPENAI_MODEL = originalOpenAIModel
    } else {
      delete process.env.OPENAI_MODEL
    }
    mock.restore()
  })

  test('routes OpenAI side queries through the OpenAI-compatible client', async () => {
    const response = await sideQuery({
      model: 'claude-sonnet-4-5',
      system: 'Explain the command safely',
      messages: [{ role: 'user', content: 'Explain rm -rf build/' }],
      tools: [
        {
          name: 'explain_command',
          description: 'Explain a command',
          input_schema: { type: 'object' },
        },
      ] as any,
      tool_choice: { type: 'tool', name: 'explain_command' } as any,
      output_format: {
        type: 'json_schema',
        schema: {
          type: 'object',
          properties: {
            riskLevel: { type: 'string' },
          },
        },
      },
      max_tokens: 256,
      stop_sequences: ['</block>'],
      querySource: 'permission_explainer',
    })

    expect(getOpenAIClient).toHaveBeenCalledTimes(1)

    expect(openAICreate).toHaveBeenCalledTimes(1)
    const [params] = openAICreate.mock.calls[0]!
    expect(params.model).toBe('gpt-5.4')
    expect(params.messages[0]).toEqual({
      role: 'system',
      content: 'Explain the command safely',
    })
    expect(params.messages[1]).toEqual({
      role: 'user',
      content: 'Explain rm -rf build/',
    })
    expect(params.tools).toEqual([
      {
        type: 'function',
        function: {
          name: 'explain_command',
          description: 'Explain a command',
          parameters: { type: 'object' },
        },
      },
    ])
    expect(params.tool_choice).toEqual({
      type: 'function',
      function: { name: 'explain_command' },
    })
    expect(params.max_completion_tokens).toBe(256)
    expect(params.stop).toEqual(['</block>'])
    expect(params.response_format).toEqual({
      type: 'json_schema',
      json_schema: {
        name: 'side_query',
        schema: {
          type: 'object',
          properties: {
            riskLevel: { type: 'string' },
          },
        },
        strict: true,
      },
    })

    expect(response.id).toBe('resp_123')
    expect(response.model).toBe('gpt-5.4')
    expect(response.stop_reason).toBe('tool_use')
    expect(response.usage.input_tokens).toBe(11)
    expect(response.usage.output_tokens).toBe(7)
    expect(response.content).toEqual([
      { type: 'text', text: 'Structured explanation ready' },
      {
        type: 'tool_use',
        id: 'call_123',
        name: 'explain_command',
        input: {
          riskLevel: 'LOW',
          explanation: 'safe',
          reasoning: 'needed',
          risk: 'none',
        },
      },
    ])
  })
})
