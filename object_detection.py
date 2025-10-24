import cv2
import threading
import tkinter as tk
from tkinter import Label, Button, Entry, Text, Frame, Checkbutton, IntVar
from PIL import Image, ImageTk
import pyttsx3
import speech_recognition as sr
from ultralytics import YOLO  # YOLOv8

# Load YOLOv8 model
MODEL_PATH = "yolov8n.pt"  # lightweight YOLOv8 model
model = YOLO(MODEL_PATH)

class VocalensYOLOQuery:
    def __init__(self, root):
        self.root = root
        self.root.title("Vocalens: Talking Lens with YOLOv8 Query")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))
        self.root.configure(bg="#121212")

        self.speech_enabled = IntVar(value=1)

        self.main_frame = Frame(root, bg="#121212")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)

        # LEFT FRAME
        self.left_frame = Frame(self.main_frame, bg="#1e1e2f", bd=2, relief="ridge")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20), pady=10)

        self.query_label = Label(self.left_frame, text="Enter your search query:", fg="white", bg="#1e1e2f", font=("Helvetica", 13))
        self.query_label.pack(pady=(20, 5))

        entry_frame = Frame(self.left_frame, bg="#1e1e2f")
        entry_frame.pack(pady=(0, 15))

        self.query_entry = Entry(entry_frame, width=34, font=("Helvetica", 13), bg="#2c2c3e", fg="white", insertbackground="white", relief="flat")
        self.query_entry.pack(side=tk.LEFT, ipady=6)

        self.mic_button = Button(entry_frame, text="🎤", command=self.speech_to_text, font=("Helvetica", 12), bg="#444", fg="white", relief="flat", width=3)
        self.mic_button.pack(side=tk.LEFT, padx=(5, 0))

        self.capture_button = Button(self.left_frame, text="Capture & Process", command=self.process_image, font=("Helvetica", 12), bg="#4CAF50", fg="white", relief="flat")
        self.capture_button.pack(pady=(0, 10), ipadx=10, ipady=5)

        self.speech_toggle = Checkbutton(self.left_frame, text="Enable automatic Speech Output", variable=self.speech_enabled, bg="#1e1e2f", fg="white", selectcolor="#1e1e2f", font=("Helvetica", 12), activebackground="#1e1e2f")
        self.speech_toggle.pack()

        self.ai_label = Label(self.left_frame, text="Detected Objects:", fg="white", bg="#1e1e2f", font=("Helvetica", 13))
        self.ai_label.pack()

        self.ai_text = Text(self.left_frame, height=18, width=55, font=("Helvetica", 12), bg="#2c2c3e", fg="white", insertbackground="white", relief="flat", wrap=tk.WORD)
        self.ai_text.pack(pady=(10, 20), padx=20)

        self.speak_button = Button(self.left_frame, text="🔊 Speak Detected Text", command=self.speak_extracted_text,
                                   font=("Helvetica", 12), bg="#4CAF50", fg="white", relief="flat")
        self.speak_button.pack(pady=(0, 20), ipadx=10, ipady=5)

        # RIGHT FRAME
        self.right_frame = Frame(self.main_frame, bg="#1e1e2f", bd=2, relief="ridge")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, pady=10)

        self.cam_label = Label(self.right_frame, bg="#000000")
        self.cam_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.latest_frame = None
        self.cam_thread = threading.Thread(target=self.start_camera_feed, daemon=True)
        self.cam_thread.start()

    def start_camera_feed(self):
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            print("❌ Could not open webcam.")
            return

        def update_feed():
            ret, frame = self.capture.read()
            if ret:
                self.latest_frame = frame.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (self.right_frame.winfo_width() or 800,
                                                   self.right_frame.winfo_height() or 600))
                img = Image.fromarray(frame_rgb)
                img = ImageTk.PhotoImage(img)
                self.cam_label.configure(image=img)
                self.cam_label.image = img
            self.root.after(10, update_feed)

        self.root.after(1000, update_feed)

    def capture_latest_frame(self):
        if self.latest_frame is None:
            print("❌ No frame captured yet.")
            return None
        return self.latest_frame

    def process_image(self):
        query = self.query_entry.get().strip().lower()
        if not query:
            print("❌ Query cannot be empty.")
            return

        frame = self.capture_latest_frame()
        if frame is None:
            return

        results = model.predict(frame)[0]  # YOLOv8 prediction
        all_detected_objects = [results.names[int(cls)] for cls in results.boxes.cls.tolist()]

        # Filter by query
        matched_objects = [obj for obj in all_detected_objects if query in obj.lower()]
        if matched_objects:
            detected_text = ", ".join(matched_objects)
        else:
            detected_text = "Sorry, couldn't identify the object."

        self.ai_text.delete("1.0", tk.END)
        self.ai_text.insert(tk.END, detected_text)

        if self.speech_enabled.get():
            self.speak_response(detected_text)

    def speak_extracted_text(self):
        text = self.ai_text.get("1.0", tk.END).strip()
        if text:
            self.speak_response(text)
        else:
            print("❌ No text to speak.")

    def speak_response(self, response_text):
        engine = pyttsx3.init()
        engine.say(response_text)
        engine.runAndWait()

    def speech_to_text(self):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            self.query_entry.delete(0, tk.END)
            print("🎙️ Listening...")
            self.query_entry.insert(0, "Listening...")
            self.root.update()
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=5)
                query = recognizer.recognize_google(audio)
                print(f"✅ You said: {query}")
                self.query_entry.delete(0, tk.END)
                self.query_entry.insert(0, query)
            except sr.WaitTimeoutError:
                print("❌ Listening timed out.")
                self.query_entry.delete(0, tk.END)
                self.query_entry.insert(0, "Listening timed out.")
            except sr.UnknownValueError:
                print("❌ Could not understand audio.")
                self.query_entry.delete(0, tk.END)
                self.query_entry.insert(0, "Couldn't recognize speech.")
            except sr.RequestError as e:
                print(f"❌ Could not request results; {e}")
                self.query_entry.delete(0, tk.END)
                self.query_entry.insert(0, "Speech service error.")

# Launch the app
root = tk.Tk()
app = VocalensYOLOQuery(root)
root.mainloop()
