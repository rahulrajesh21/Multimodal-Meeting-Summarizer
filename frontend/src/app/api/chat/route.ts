import { StreamingTextResponse, StreamData } from 'ai';

export async function POST(req: Request) {
  const { messages, model, llm_backend } = await req.json();
  
  if (!messages || messages.length === 0) {
    return new Response('No messages', { status: 400 });
  }

  const lastMessage = messages[messages.length - 1].content;
  const history = messages.slice(0, -1).map((m: any) => ({
    role: m.role,
    content: m.content
  }));

  const backendUrl = process.env.API_BASE || 'http://127.0.0.1:8000';

  const pyRes = await fetch(`${backendUrl}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: lastMessage, history, model, llm_backend: llm_backend || 'lmstudio' }),
  });

  if (!pyRes.ok) {
    return new Response(await pyRes.text(), { status: pyRes.status });
  }

  const data = new StreamData();

  const stream = new ReadableStream({
    async start(controller) {
      const reader = pyRes.body?.getReader();
      if (!reader) {
        controller.close();
        await data.close();
        return;
      }

      const decoder = new TextDecoder();
      let buffer = '';

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const dataStr = line.slice(6).trim();
              if (dataStr === '[DONE]') continue;
              const json = JSON.parse(dataStr);
              
              if (json.type === 'step') {
                data.append({ type: 'step', step: json.step });
              } else if (json.type === 'reply') {
                if (json.content) controller.enqueue(json.content);
                data.append({ type: 'model', model: json.model });
                if (json.steps && json.steps.length > 0) {
                  json.steps.forEach((step: any) => {
                    data.append({ type: 'step', step });
                  });
                }
              } else if (json.type === 'error') {
                console.error("Backend error:", json.message);
              }
            }
          }
        }
        
        if (buffer && buffer.startsWith('data: ')) {
          const dataStr = buffer.slice(6).trim();
          if (dataStr !== '[DONE]') {
            const json = JSON.parse(dataStr);
            if (json.type === 'reply' && json.content) {
              controller.enqueue(json.content);
            }
          }
        }
      } catch (err) {
        console.error('Error parsing stream:', err);
      } finally {
        reader.releaseLock();
        controller.close();
        await data.close();
      }
    }
  });

  return new StreamingTextResponse(stream, {}, data);
}
