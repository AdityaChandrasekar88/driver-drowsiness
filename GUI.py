import cv2
import numpy as np
from scipy.spatial import distance as dist
import dlib
from pygame import mixer
import time
import threading
import os
import sqlite3
from tkinter import *
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# Initialize mixer and load alarm sound
mixer.init()
sound = mixer.Sound('alarm.wav') if os.path.exists('alarm.wav') else None

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Database setup
def init_db():
    with sqlite3.connect('drowsiness.db') as conn:
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS users
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     username TEXT UNIQUE,
                     password TEXT,
                     role TEXT,
                     fullname TEXT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS sessions
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     driver_id INTEGER,
                     start_time TIMESTAMP,
                     end_time TIMESTAMP,
                     max_score INTEGER,
                     avg_score REAL,
                     FOREIGN KEY(driver_id) REFERENCES users(id))''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS session_data
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     session_id INTEGER,
                     timestamp TIMESTAMP,
                     score INTEGER,
                     ear REAL,
                     mar REAL,
                     FOREIGN KEY(session_id) REFERENCES sessions(id))''')
        
        # Create admin if not exists
        c.execute("SELECT * FROM users WHERE username='admin'")
        if not c.fetchone():
            c.execute("INSERT INTO users (username, password, role, fullname) VALUES (?, ?, ?, ?)",
                     ('admin', 'admin123', 'admin', 'Administrator'))
        conn.commit()

init_db()

# Drowsiness Detection Functions
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C != 0 else 0

def mouth_aspect_ratio(mouth):
    # Calculate distances between key mouth points
    A = dist.euclidean(mouth[0], mouth[6])  # Horizontal distance
    
    # Vertical distances
    B1 = dist.euclidean(mouth[2], mouth[10])  # Top to bottom center
    B2 = dist.euclidean(mouth[4], mouth[8])   # Midpoints
    
    # Additional vertical measurement
    C = dist.euclidean(mouth[3], mouth[9])    # Center points
    
    # Combined mouth aspect ratio
    mar = (B1 + B2 + C) / (3.0 * A) if A != 0 else 0
    return mar

def play_short_alarm():
    if sound:
        sound.play()
        time.sleep(0.5)
        sound.stop()

# Constants
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.85  # Increased threshold for better accuracy
EAR_CONSEC_FRAMES = 20
YAWN_CONSEC_FRAMES = 12  # Reduced frames for quicker yawn detection
SCORE_THRESHOLD = 15
ALARM_COOLDOWN = 2
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))

class DrowsinessDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drowsiness Detection System")
        self.root.geometry("1000x700")
        self.current_user = None
        self.current_session_id = None
        self.detection_active = False
        self.show_login_screen()
    
    def clear_frame(self):
        for widget in self.root.winfo_children():
            widget.destroy()
    
    def show_login_screen(self):
        self.clear_frame()
        
        Label(self.root, text="Drowsiness Detection System", font=('Helvetica', 16)).pack(pady=20)
        
        self.role_var = StringVar(value="driver")
        Label(self.root, text="Select Role:").pack()
        Radiobutton(self.root, text="Driver", variable=self.role_var, value="driver").pack()
        Radiobutton(self.root, text="Admin", variable=self.role_var, value="admin").pack()
        
        Label(self.root, text="Username:").pack(pady=(10,0))
        self.username_entry = Entry(self.root)
        self.username_entry.pack()
        
        Label(self.root, text="Password:").pack(pady=(10,0))
        self.password_entry = Entry(self.root, show="*")
        self.password_entry.pack()
        
        Button(self.root, text="Login", command=self.login).pack(pady=20)
        
        self.login_status = Label(self.root, text="", fg="red")
        self.login_status.pack()
    
    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        role = self.role_var.get()
        
        with sqlite3.connect('drowsiness.db') as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE username=? AND role=?", (username, role))
            user = c.fetchone()
            
        if user and user[2] == password:
            self.current_user = {
                'id': user[0],
                'username': user[1],
                'role': user[3],
                'fullname': user[4]
            }
            if role == "admin":
                self.show_admin_dashboard()
            else:
                self.show_driver_dashboard()
        else:
            self.login_status.config(text="Invalid username or password")
    
    def show_admin_dashboard(self):
        self.clear_frame()
        
        header_frame = Frame(self.root)
        header_frame.pack(fill=X, pady=10)
        
        Label(header_frame, text=f"Admin Dashboard - Welcome {self.current_user['fullname']}", 
              font=('Helvetica', 14)).pack(side=LEFT, padx=10)
        
        Button(header_frame, text="Logout", command=self.show_login_screen).pack(side=RIGHT, padx=10)
        
        tab_control = ttk.Notebook(self.root)
        
        driver_tab = Frame(tab_control)
        tab_control.add(driver_tab, text="Manage Drivers")
        
        # Add New Driver Frame
        add_frame = Frame(driver_tab)
        add_frame.grid(row=0, column=0, columnspan=2, pady=10)
        
        Label(add_frame, text="Add New Driver", font=('Helvetica', 12)).pack()
        
        Label(add_frame, text="Full Name:").pack(pady=(5,0))
        self.driver_name_entry = Entry(add_frame)
        self.driver_name_entry.pack()
        
        Label(add_frame, text="Username:").pack(pady=(5,0))
        self.driver_username_entry = Entry(add_frame)
        self.driver_username_entry.pack()
        
        Label(add_frame, text="Password:").pack(pady=(5,0))
        self.driver_password_entry = Entry(add_frame, show="*")
        self.driver_password_entry.pack()
        
        Button(add_frame, text="Add Driver", command=self.add_driver).pack(pady=10)
        
        # Driver List with Remove Button
        list_frame = Frame(driver_tab)
        list_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
        
        self.driver_tree = ttk.Treeview(list_frame, columns=('id', 'username', 'fullname'), show='headings')
        self.driver_tree.heading('id', text='ID')
        self.driver_tree.heading('username', text='Username')
        self.driver_tree.heading('fullname', text='Full Name')
        self.driver_tree.pack(side=LEFT, fill=BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.driver_tree.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.driver_tree.configure(yscrollcommand=scrollbar.set)
        
        # Remove Driver Button
        remove_frame = Frame(driver_tab)
        remove_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        Button(remove_frame, text="Remove Selected Driver", command=self.remove_driver).pack()
        
        self.load_drivers()
        
        reports_tab = Frame(tab_control)
        tab_control.add(reports_tab, text="Driver Reports")
        
        selection_frame = Frame(reports_tab)
        selection_frame.pack(fill=X, pady=5)
        
        Label(selection_frame, text="Select Driver:").pack(side=LEFT, padx=5)
        self.driver_var = StringVar()
        self.driver_dropdown = ttk.Combobox(selection_frame, textvariable=self.driver_var)
        self.driver_dropdown.pack(side=LEFT, padx=5)
        
        Button(selection_frame, text="Show Sessions", command=self.load_sessions).pack(side=LEFT, padx=5)
        
        self.sessions_tree = ttk.Treeview(reports_tab, 
                                        columns=('id', 'driver', 'start', 'end', 'max_score', 'avg_score', 'last_score'), 
                                        show='headings')
        self.sessions_tree.heading('id', text='Session ID')
        self.sessions_tree.heading('driver', text='Driver')
        self.sessions_tree.heading('start', text='Start Time')
        self.sessions_tree.heading('end', text='End Time')
        self.sessions_tree.heading('max_score', text='Max Score')
        self.sessions_tree.heading('avg_score', text='Avg Score')
        self.sessions_tree.heading('last_score', text='Last Score')
        
        self.sessions_tree.column('id', width=80)
        self.sessions_tree.column('driver', width=120)
        self.sessions_tree.column('start', width=150)
        self.sessions_tree.column('end', width=150)
        self.sessions_tree.column('max_score', width=80)
        self.sessions_tree.column('avg_score', width=80)
        self.sessions_tree.column('last_score', width=80)
        
        self.sessions_tree.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        self.load_driver_dropdown()
        
        tab_control.pack(expand=1, fill="both")
    
    def remove_driver(self):
        selected_item = self.driver_tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "Please select a driver to remove")
            return
        
        driver_id = self.driver_tree.item(selected_item)['values'][0]
        driver_name = self.driver_tree.item(selected_item)['values'][2]
        
        confirm = messagebox.askyesno("Confirm", f"Are you sure you want to remove {driver_name}?")
        if not confirm:
            return
        
        try:
            with sqlite3.connect('drowsiness.db') as conn:
                c = conn.cursor()
                
                # First delete related session data
                c.execute("DELETE FROM session_data WHERE session_id IN (SELECT id FROM sessions WHERE driver_id = ?)", 
                          (driver_id,))
                # Then delete sessions
                c.execute("DELETE FROM sessions WHERE driver_id = ?", (driver_id,))
                # Finally delete the driver
                c.execute("DELETE FROM users WHERE id = ?", (driver_id,))
                
                conn.commit()
            
            messagebox.showinfo("Success", "Driver removed successfully")
            self.load_drivers()
            self.load_driver_dropdown()  # Refresh the dropdown in reports tab
            
        except sqlite3.Error as e:
            messagebox.showerror("Error", f"Failed to remove driver: {e}")
    
    def add_driver(self):
        fullname = self.driver_name_entry.get()
        username = self.driver_username_entry.get()
        password = self.driver_password_entry.get()
        
        if not all([fullname, username, password]):
            messagebox.showerror("Error", "All fields are required")
            return
        
        try:
            with sqlite3.connect('drowsiness.db') as conn:
                c = conn.cursor()
                c.execute("INSERT INTO users (username, password, role, fullname) VALUES (?, ?, ?, ?)",
                         (username, password, 'driver', fullname))
                conn.commit()
            messagebox.showinfo("Success", "Driver added successfully")
            self.load_drivers()
            self.load_driver_dropdown()
            self.driver_name_entry.delete(0, END)
            self.driver_username_entry.delete(0, END)
            self.driver_password_entry.delete(0, END)
        except sqlite3.IntegrityError:
            messagebox.showerror("Error", "Username already exists")
    
    def load_drivers(self):
        with sqlite3.connect('drowsiness.db') as conn:
            c = conn.cursor()
            for item in self.driver_tree.get_children():
                self.driver_tree.delete(item)
            c.execute("SELECT id, username, fullname FROM users WHERE role='driver'")
            for row in c.fetchall():
                self.driver_tree.insert('', 'end', values=row)
    
    def load_driver_dropdown(self):
        with sqlite3.connect('drowsiness.db') as conn:
            c = conn.cursor()
            c.execute("SELECT id, fullname FROM users WHERE role='driver'")
            drivers = c.fetchall()
            self.driver_dropdown['values'] = [f"{id} - {name}" for id, name in drivers]
            if drivers:
                self.driver_var.set(self.driver_dropdown['values'][0])
    
    def load_sessions(self):
        driver_info = self.driver_var.get()
        if not driver_info:
            return
        
        try:
            driver_id = int(driver_info.split(" - ")[0])
        except (ValueError, IndexError):
            return
        
        with sqlite3.connect('drowsiness.db') as conn:
            c = conn.cursor()
            for item in self.sessions_tree.get_children():
                self.sessions_tree.delete(item)
            
            c.execute('''SELECT s.id, u.fullname, s.start_time, s.end_time, s.max_score, s.avg_score,
                        (SELECT sd.score FROM session_data sd 
                         WHERE sd.session_id = s.id 
                         ORDER BY sd.timestamp DESC LIMIT 1) as last_score
                        FROM sessions s 
                        JOIN users u ON s.driver_id = u.id 
                        WHERE s.driver_id=? 
                        ORDER BY s.start_time DESC''', (driver_id,))
            
            for row in c.fetchall():
                self.sessions_tree.insert('', 'end', values=row)
    
    def show_driver_dashboard(self):
        self.clear_frame()
        
        header_frame = Frame(self.root)
        header_frame.pack(fill=X, pady=10)
        
        Label(header_frame, text=f"Driver Dashboard - Welcome {self.current_user['fullname']}", 
              font=('Helvetica', 14)).pack(side=LEFT, padx=10)
        
        Button(header_frame, text="Logout", command=self.show_login_screen).pack(side=RIGHT, padx=10)
        
        self.detection_frame = Frame(self.root)
        self.detection_frame.pack(pady=20)
        
        self.video_label = Label(self.detection_frame)
        self.video_label.pack()
        
        button_frame = Frame(self.root)
        button_frame.pack(pady=10)
        
        self.start_button = Button(button_frame, text="Start Detection", command=self.start_detection)
        self.start_button.pack(side=LEFT, padx=20)
        
        self.stop_button = Button(button_frame, text="Stop Detection", command=self.stop_detection, state=DISABLED)
        self.stop_button.pack(side=LEFT, padx=20)
        
        self.status_frame = Frame(self.root)
        self.status_frame.pack(pady=10)
        
        self.eye_status_label = Label(self.status_frame, text="Eye Status: -", font=('Helvetica', 10))
        self.eye_status_label.grid(row=0, column=0, padx=10)
        
        self.mouth_status_label = Label(self.status_frame, text="Mouth Status: -", font=('Helvetica', 10))
        self.mouth_status_label.grid(row=0, column=1, padx=10)
        
        self.overall_status_label = Label(self.status_frame, text="Overall Status: -", font=('Helvetica', 10))
        self.overall_status_label.grid(row=0, column=2, padx=10)
        
        self.score_label = Label(self.root, text="Score: 0", font=('Helvetica', 12))
        self.score_label.pack()
        
        Label(self.root, text="Recent Sessions", font=('Helvetica', 12)).pack(pady=(20,5))
        
        self.session_tree = ttk.Treeview(self.root, columns=('id', 'start', 'end', 'max_score', 'avg_score'), show='headings', height=5)
        self.session_tree.heading('id', text='Session ID')
        self.session_tree.heading('start', text='Start Time')
        self.session_tree.heading('end', text='End Time')
        self.session_tree.heading('max_score', text='Max Score')
        self.session_tree.heading('avg_score', text='Avg Score')
        self.session_tree.pack(pady=10)
        
        self.load_session_history()
    
    def load_session_history(self):
        with sqlite3.connect('drowsiness.db') as conn:
            c = conn.cursor()
            for item in self.session_tree.get_children():
                self.session_tree.delete(item)
            c.execute('''SELECT id, start_time, end_time, max_score, avg_score 
                        FROM sessions WHERE driver_id=? ORDER BY start_time DESC LIMIT 5''', 
                        (self.current_user['id'],))
            for row in c.fetchall():
                self.session_tree.insert('', 'end', values=row)
    
    def start_detection(self):
        if not self.detection_active:
            self.detection_active = True
            self.start_button.config(state=DISABLED)
            self.stop_button.config(state=NORMAL)
            
            with sqlite3.connect('drowsiness.db') as conn:
                c = conn.cursor()
                c.execute("INSERT INTO sessions (driver_id, start_time) VALUES (?, datetime('now'))", 
                         (self.current_user['id'],))
                self.current_session_id = c.lastrowid
                conn.commit()
            
            self.cap = cv2.VideoCapture(0)
            self.eye_frame_counter = 0
            self.yawn_frame_counter = 0
            self.score = 0
            self.last_alarm_time = 0
            self.avg_ear = 0.0
            self.avg_mar = 0.0
            self.scores = []
            self.update_detection()
    
    def stop_detection(self):
        if self.detection_active:
            self.detection_active = False
            if hasattr(self, 'start_button') and self.start_button.winfo_exists():
                self.start_button.config(state=NORMAL)
            if hasattr(self, 'stop_button') and self.stop_button.winfo_exists():
                self.stop_button.config(state=DISABLED)
            
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
                del self.cap
            
            if self.current_session_id:
                with sqlite3.connect('drowsiness.db') as conn:
                    c = conn.cursor()
                    avg_score = sum(self.scores)/len(self.scores) if self.scores else 0
                    c.execute('''UPDATE sessions SET 
                                end_time = datetime('now'),
                                max_score = ?,
                                avg_score = ?
                                WHERE id = ?''',
                             (max(self.scores) if self.scores else 0, avg_score, self.current_session_id))
                    conn.commit()
                
                if hasattr(self, 'session_tree') and self.session_tree.winfo_exists():
                    self.load_session_history()
                self.current_session_id = None
    
    def update_detection(self):
        if not self.detection_active or not hasattr(self, 'cap') or not self.video_label.winfo_exists():
            return
            
        try:
            ret, frame = self.cap.read()
            if not ret:
                self.root.after(10, self.update_detection)
                return
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            
            eye_status = "Open"
            mouth_status = "Closed"
            overall_status = "Awake"
            
            if len(faces) > 0:
                for face in faces:
                    shape = predictor(gray, face)
                    shape = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
                    
                    left_eye = shape[LEFT_EYE]
                    right_eye = shape[RIGHT_EYE]
                    mouth = shape[MOUTH]
                    
                    # Calculate aspect ratios
                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    self.avg_ear = (left_ear + right_ear) / 2.0
                    self.avg_mar = mouth_aspect_ratio(mouth)
                    
                    # Draw landmarks with different colors for open/closed states
                    eye_color = (0, 255, 0)  # Green for open eyes
                    mouth_color = (0, 255, 0)  # Green for closed mouth
                    
                    # Eye status detection
                    if self.avg_ear < EAR_THRESHOLD:
                        self.eye_frame_counter += 1
                        eye_status = "Closed"
                        eye_color = (0, 0, 255)  # Red for closed eyes
                    else:
                        self.eye_frame_counter = max(0, self.eye_frame_counter - 1)
                    
                    # Enhanced mouth status detection
                    if self.avg_mar > MAR_THRESHOLD:
                        self.yawn_frame_counter += 1
                        mouth_status = "Yawning"
                        mouth_color = (0, 0, 255)  # Red for open mouth
                        
                        # Additional visual feedback for yawning
                        cv2.line(frame, tuple(mouth[2]), tuple(mouth[10]), (0, 0, 255), 2)
                        cv2.line(frame, tuple(mouth[4]), tuple(mouth[8]), (0, 0, 255), 2)
                    else:
                        self.yawn_frame_counter = max(0, self.yawn_frame_counter - 1)
                    
                    # Draw landmarks with status-based colors
                    cv2.polylines(frame, [left_eye], True, eye_color, 1)
                    cv2.polylines(frame, [right_eye], True, eye_color, 1)
                    cv2.polylines(frame, [mouth], True, mouth_color, 1)
                    
                    # Update score based on both eye and mouth status
                    if eye_status == "Closed" and self.eye_frame_counter >= EAR_CONSEC_FRAMES:
                        self.score += 1
                    elif mouth_status == "Yawning" and self.yawn_frame_counter >= YAWN_CONSEC_FRAMES:
                        self.score += 2  # Higher weight for yawning
                    else:
                        self.score = max(0, self.score - 1)
                    
                    self.scores.append(self.score)
                    
                    if self.score > SCORE_THRESHOLD:
                        overall_status = "Drowsy"
                    
                    current_time = time.time()
                    if overall_status == "Drowsy" and (current_time - self.last_alarm_time) > ALARM_COOLDOWN:
                        threading.Thread(target=play_short_alarm, daemon=True).start()
                        self.last_alarm_time = current_time
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                self.score = max(0, self.score - 1)
                self.eye_frame_counter = 0
                self.yawn_frame_counter = 0
                self.avg_ear = 0.0
                self.avg_mar = 0.0
            
            if self.video_label.winfo_exists():
                eye_color = "red" if eye_status == "Closed" else "green"
                mouth_color = "red" if mouth_status == "Yawning" else "green"
                overall_color = "red" if overall_status == "Drowsy" else "green"
                
                if hasattr(self, 'eye_status_label') and self.eye_status_label.winfo_exists():
                    self.eye_status_label.config(text=f"Eye Status: {eye_status}", fg=eye_color)
                if hasattr(self, 'mouth_status_label') and self.mouth_status_label.winfo_exists():
                    self.mouth_status_label.config(text=f"Mouth Status: {mouth_status}", fg=mouth_color)
                if hasattr(self, 'overall_status_label') and self.overall_status_label.winfo_exists():
                    self.overall_status_label.config(text=f"Overall Status: {overall_status}", fg=overall_color)
                if hasattr(self, 'score_label') and self.score_label.winfo_exists():
                    self.score_label.config(text=f"Score: {self.score}")
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
                if self.current_session_id:
                    with sqlite3.connect('drowsiness.db') as conn:
                        c = conn.cursor()
                        c.execute('''INSERT INTO session_data 
                                    (session_id, timestamp, score, ear, mar) 
                                    VALUES (?, datetime('now'), ?, ?, ?)''',
                                 (self.current_session_id, self.score, self.avg_ear, self.avg_mar))
                        conn.commit()
            
            self.root.after(10, self.update_detection)
        except Exception as e:
            print(f"Error in update_detection: {e}")
            self.stop_detection()

if __name__ == "__main__":
    if not os.path.exists('shape_predictor_68_face_landmarks.dat'):
        print("Error: Please download shape_predictor_68_face_landmarks.dat")
        print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        exit()
    
    root = Tk()
    app = DrowsinessDetectionApp(root)
    root.mainloop()
    