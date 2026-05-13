import speech_recognition as sr
import subprocess
import os
import time
import asyncio
import edge_tts

# =========================
# EDGE TTS CONFIG
# =========================

VOICE = "vi-VN-HoaiMyNeural"

# =========================
# TEXT TO SPEECH
# =========================

async def async_speak(text):

    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE,

        # tốc độ nói
        rate="+15%",

        # tăng độ cao giọng
        pitch="+20Hz"
    )

    await communicate.save("voice.mp3")

    # phát audio
    os.system("start /min wmplayer voice.mp3")

    # đợi nói gần xong
    time.sleep(
        max(3, len(text) * 0.06)
    )

def speak(text):

    print("Layla:", text)

    asyncio.run(
        async_speak(text)
    )

# =========================
# SPEECH TO TEXT
# =========================

def listen():

    r = sr.Recognizer()

    with sr.Microphone() as source:

        print("Đang nghe...")

        # lọc nhiễu
        r.adjust_for_ambient_noise(
            source,
            duration=1
        )

        # độ nhạy mic
        r.energy_threshold = 250

        r.dynamic_energy_threshold = False

        try:

            audio = r.listen(
                source,
                timeout=5,
                phrase_time_limit=5
            )

            text = r.recognize_google(
                audio,
                language="vi-VN"
            )

            print("Bạn nói:", text)

            return text.lower()

        except sr.WaitTimeoutError:

            print("Không nghe thấy gì")
            return ""

        except sr.UnknownValueError:

            print("Không nhận diện được giọng nói")
            return ""

        except Exception as e:

            print("Lỗi:", e)
            return ""

# =========================
# HELPERS
# =========================

def get_existing_persons():

    predata_dir = "predata"

    if not os.path.exists(predata_dir):
        return []

    return [
        d for d in os.listdir(predata_dir)
        if os.path.isdir(
            os.path.join(predata_dir, d)
        )
    ]

def clean_name(text: str):

    for ch in ",.?;:!":
        text = text.replace(ch, "")

    return text.strip().replace(" ", "_")

# =========================
# MAIN MENU
# =========================

def main():

    speak(
        "Chào bạn, mình là Layla. "
        "Hôm nay bạn muốn làm gì? "
        "Một là lấy thông tin lưu trữ. "
        "Hai là nhận diện khuôn mặt."
    )

    text = listen()

    # =========================
    # OPTION 1
    # =========================

    if (
        "một" in text
        or "1" in text
        or "lưu trữ" in text
        or "thông tin" in text
    ):

        speak(
            "Bạn muốn ghi mới "
            "hay thêm vào thư viện?"
        )

        sub_text = listen()

        # -------------------------
        # GHI MỚI
        # -------------------------

        if (
            "mới" in sub_text
            or "ghi mới" in sub_text
            or "thêm mới" in sub_text
        ):

            speak(
                "Đang mở hệ thống lưu trữ ghi mới"
            )

            subprocess.run([
                "python",
                "main.py"
            ])

        # -------------------------
        # ENRICH
        # -------------------------

        elif (
            "thêm" in sub_text
            or "thư viện" in sub_text
            or "làm giàu" in sub_text
            or "bổ sung" in sub_text
        ):

            persons = get_existing_persons()

            if not persons:

                speak(
                    "Chưa có người nào trong thư viện. "
                    "Chuyển sang ghi mới."
                )

                subprocess.run([
                    "python",
                    "main.py"
                ])

                return

            persons_text = ", ".join([
                p.replace("_", " ")
                for p in persons
            ])

            speak(
                f"Danh sách người đã lưu gồm "
                f"{persons_text}. "
                f"Bạn muốn thêm ai?"
            )

            name_raw = listen()

            if not name_raw:

                speak(
                    "Tôi không nghe rõ tên. "
                    "Vui lòng thử lại."
                )

                return

            name = clean_name(name_raw)

            speak(
                f"Đang thêm dữ liệu cho "
                f"{name_raw}"
            )

            subprocess.run([
                "python",
                "main.py",
                "--enrich",
                name
            ])

        else:

            speak(
                "Tôi chưa hiểu yêu cầu của bạn"
            )

    # =========================
    # OPTION 2
    # =========================

    elif (
        "hai" in text
        or "2" in text
        or "nhận diện" in text
        or "khuôn mặt" in text
    ):

        speak(
            "Đang mở hệ thống nhận diện khuôn mặt"
        )

        subprocess.run([
            "python",
            "04_recognize.py"
        ])

    # =========================
    # UNKNOWN
    # =========================

    else:

        speak(
            "Tôi chưa hiểu yêu cầu của bạn"
        )

    time.sleep(2)

# =========================

if __name__ == "__main__":
    main()