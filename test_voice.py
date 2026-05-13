import asyncio
import edge_tts

async def test():

    communicate = edge_tts.Communicate(
        text="Xin chào mình là Layla",
        voice="vi-VN-HoaiMyNeural"
    )

    await communicate.save("test.mp3")

asyncio.run(test())
