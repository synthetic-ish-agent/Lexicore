import asyncio
import base64
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
import uvicorn

# --- CONFIGURATION ---
# Using the key you provided
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "AIzaSyAYgL7efFFHnFmtaeRsQngTvKeA-dzoIUg"
MODEL = "gemini-2.5-flash"

app = FastAPI()

@app.websocket("/voice")
async def voice_endpoint(ws: WebSocket):
    await ws.accept()
    print("✅ [SERVER] Streamlit connected to WebSocket")
    
    client = genai.Client(
        api_key=GOOGLE_API_KEY, 
        http_options={"api_version": "v1alpha"}
    )

    try:
        async with client.aio.live.connect(
            model=MODEL,
            config={
                "response_modalities": ["AUDIO"],
                "system_instruction": "You are LexiCore, a forensic theological debater. Respond with clarity and authority.",
                "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}}
            }
        ) as session:
            print("✅ [SERVER] Connected to Gemini Live API")
            
            async def gemini_to_ws():
                try:
                    async for msg in session.receive():
                        if msg.server_content and msg.server_content.model_turn:
                            for part in msg.server_content.model_turn.parts:
                                if part.inline_data:
                                    print(f"🔊 [SERVER] Gemini sent {len(part.inline_data.data)} bytes of audio")
                                    await ws.send_text(base64.b64encode(part.inline_data.data).decode())
                except Exception as e:
                    print(f"❌ [SERVER] Gemini output error: {e}")

            task = asyncio.create_task(gemini_to_ws())
            
            try:
                while True:
                    data = await ws.receive_bytes()
                    if data:
                        # This log is vital: if you don't see this, the mic isn't sending
                        print(f"🎤 [SERVER] Received {len(data)} bytes from UI")
                        
                        await session.send_realtime_input(
                            audio={"data": data, "mime_type": "audio/pcm;rate=16000"}
                        )
            except WebSocketDisconnect:
                print("ℹ️ [SERVER] Streamlit UI closed the connection.")
            finally:
                task.cancel()
                
    except Exception as e:
        print(f"❌ [SERVER] Connection failed: {e}")
        # Send a short reason to avoid the control frame error
        await ws.close(code=1011, reason="Check server logs")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)