import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from db_handler import DatabaseHandler

# Load saved model and label encoder
model = joblib.load('asl_random_forest_model.joblib')
label_encoder = joblib.load('asl_label_encoder.joblib')

# Function to normalize landmarks (must match training)
def normalize_landmarks_3d(landmarks):
    x_wrist, y_wrist, z_wrist = landmarks[0]
    x_tip, y_tip, z_tip = landmarks[8]

    scale = np.sqrt((x_tip - x_wrist)**2 + (y_tip - y_wrist)**2 + (z_tip - z_wrist)**2)
    if scale == 0:
        return np.zeros(63)

    normalized = []
    for x, y, z in landmarks:
        normalized.append((x - x_wrist) / scale)
        normalized.append((y - y_wrist) / scale)
        normalized.append((z - z_wrist) / scale)
    return np.array(normalized)

class ASLDetectorGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("ASL Letter Detector")
        self.window.geometry("800x600")
        
        # Initialize database handler
        self.db = DatabaseHandler()
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Create GUI components
        self.frame_video = ttk.Frame(window)
        self.frame_video.pack(pady=10)
        
        self.lbl_video = ttk.Label(self.frame_video)
        self.lbl_video.pack()
        
        self.frame_result = ttk.Frame(window)
        self.frame_result.pack(pady=20)
        
        self.lbl_result_title = ttk.Label(self.frame_result, text="Detected Letter:", font=("Arial", 14))
        self.lbl_result_title.pack()
        
        self.lbl_result = ttk.Label(self.frame_result, text="None", font=("Arial", 72, "bold"))
        self.lbl_result.pack()
        
        # Create a frame for buttons
        self.frame_buttons = ttk.Frame(window)
        self.frame_buttons.pack(pady=10)
        
        self.btn_start = ttk.Button(self.frame_buttons, text="Start Camera", command=self.start_camera)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = ttk.Button(self.frame_buttons, text="Stop Camera", command=self.stop_camera, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        
        self.btn_save = ttk.Button(self.frame_buttons, text="Save Letter", command=self.save_current_letter, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=5)
        
        self.btn_view_db = ttk.Button(self.frame_buttons, text="View Records", command=self.view_database_records)
        self.btn_view_db.pack(side=tk.LEFT, padx=5)
        
        # Add a frame for displaying saved letters
        self.frame_saved = ttk.Frame(window)
        self.frame_saved.pack(pady=10)
        
        self.lbl_saved_title = ttk.Label(self.frame_saved, text="Recently Saved Letters:", font=("Arial", 12))
        self.lbl_saved_title.pack()
        
        self.lbl_saved_letters = ttk.Label(self.frame_saved, text="", font=("Arial", 10))
        self.lbl_saved_letters.pack()
        
        # Camera setup
        self.camera = None
        self.is_running = False
        self.current_letter = "None"
        
    def view_database_records(self):
        # Create a new window for displaying all records
        records_window = tk.Toplevel(self.window)
        records_window.title("ASL Letter Records")
        records_window.geometry("500x400")
        
        # Create a frame for the treeview
        frame = ttk.Frame(records_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a scrollbar
        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create treeview
        columns = ("ID", "Letter", "Timestamp")
        treeview = ttk.Treeview(frame, columns=columns, show="headings", yscrollcommand=scrollbar.set)
        
        # Configure scrollbar
        scrollbar.config(command=treeview.yview)
        
        # Set column headings
        treeview.heading("ID", text="ID")
        treeview.heading("Letter", text="Letter")
        treeview.heading("Timestamp", text="Timestamp")
        
        # Set column widths
        treeview.column("ID", width=50)
        treeview.column("Letter", width=100)
        treeview.column("Timestamp", width=150)
        
        # Get all records from database
        records = self.db.get_saved_letters()
        
        # Insert records into treeview
        for record in records:
            treeview.insert("", tk.END, values=record)
        
        treeview.pack(fill=tk.BOTH, expand=True)
        
        # Add a refresh button
        refresh_btn = ttk.Button(records_window, text="Refresh", 
                                command=lambda: self.refresh_records_view(treeview))
        refresh_btn.pack(pady=10)
    
    def refresh_records_view(self, treeview):
        # Clear existing items
        for item in treeview.get_children():
            treeview.delete(item)
        
        # Get updated records
        records = self.db.get_saved_letters()
        
        # Insert records into treeview
        for record in records:
            treeview.insert("", tk.END, values=record)
        
    def save_current_letter(self):
        if self.current_letter != "None" and self.current_letter != "No hand detected":
            try:
                self.db.save_letter(self.current_letter)
                self.update_saved_letters_display()
                messagebox.showinfo("Success", f"Letter '{self.current_letter}' saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save letter: {str(e)}")
    
    def update_saved_letters_display(self):
        saved_letters = self.db.get_saved_letters()[:5]  # Get last 5 saved letters
        if saved_letters:
            display_text = "Last 5 saved letters:\n" + "\n".join(
                f"{letter[1]} - {letter[2]}" for letter in saved_letters
            )
        else:
            display_text = "No letters saved yet"
        self.lbl_saved_letters.config(text=display_text)
        
    def start_camera(self):
        if not self.is_running:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                self.lbl_result.config(text="Error: Cannot open camera")
                return
                
            self.is_running = True
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.btn_save.config(state=tk.NORMAL)
            self.update_frame()
    
    def stop_camera(self):
        if self.is_running:
            self.is_running = False
            if self.camera:
                self.camera.release()
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            self.btn_save.config(state=tk.DISABLED)
            self.lbl_result.config(text="None")
            self.current_letter = "None"
    
    def predict_letter(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(image_rgb)
        
        # Draw hand landmarks on the frame
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
            # Get landmarks for prediction
            landmarks = [(lm.x, lm.y, lm.z) for lm in result.multi_hand_landmarks[0].landmark]
            # Normalize and reshape input
            normalized = normalize_landmarks_3d(landmarks).reshape(1, -1)
            # Build a DataFrame with the same columns the model saw
            df_input = pd.DataFrame(normalized, columns=model.feature_names_in_)
            pred_idx = model.predict(df_input)[0]
            predicted_letter = label_encoder.inverse_transform([pred_idx])[0]
            self.current_letter = predicted_letter
            return predicted_letter, frame
        else:
            self.current_letter = "No hand detected"
            return "No hand detected", frame
    
    def update_frame(self):
        if self.is_running:
            ret, frame = self.camera.read()
            if ret:
                # Flip the frame horizontally for a more natural mirror view
                frame = cv2.flip(frame, 1)
                
                # Predict letter from current frame
                letter, annotated_frame = self.predict_letter(frame)
                self.lbl_result.config(text=letter)
                
                # Convert OpenCV frame to tkinter compatible image
                img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = ImageTk.PhotoImage(image=img)
                
                # Update the video label
                self.lbl_video.config(image=img)
                self.lbl_video.image = img
                
            # Schedule the next update
            self.window.after(10, self.update_frame)
    
    def __del__(self):
        if self.camera and self.camera.isOpened():
            self.camera.release()

# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = ASLDetectorGUI(root)
    root.mainloop() 