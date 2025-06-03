from flask import Flask, render_template, request, redirect, jsonify, session
import os
import cv2
import numpy as np
import base64
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
from werkzeug.security import generate_password_hash
import json 
from flask import Response
from firebase_admin import credentials, initialize_app, db, storage
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from collections import defaultdict
import re  # Tambahkan baris ini
from flask import Flask, render_template, request, redirect, session, jsonify
# ... import lainnya yang sudah ada
from tensorflow.keras.models import load_model
from collections import defaultdict
from datetime import datetime
from flask import Flask, render_template, request
import firebase_admin
from firebase_admin import credentials, db
import re  # Tambahkan baris ini
from flask import Flask, render_template, request, redirect, session, jsonify
# ... import lainnya yang sudah ada


# Inisialisasi Firebase
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Ganti dengan secret key yang aman
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # Maksimum 64 MB

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
dataset_path = 'DataSet'
from firebase_admin import credentials, initialize_app, storage


# Load model hasil fine-tuning
model = tf.keras.models.load_model('models/best_finetuned_model_mobilenet.keras')


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Ganti dengan secret key yang aman
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # Maksimum 64 MB

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
dataset_path = 'DataSet'


cred = credentials.Certificate("D:/coba/facerecognition-c8264-firebase-adminsdk-nodyk-90850d2e73.json")

initialize_app(cred, {
    'databaseURL': 'https://facerecognition-c8264-default-rtdb.firebaseio.com/',
    'storageBucket': 'facerecognition-c8264.appspot.com'
})

bucket = storage.bucket()



# Load label dari file JSON
with open('label_map.json', 'r') as f:
    labels = json.load(f)

# Path dataset
dataset_path = "D:/cobaf/AttendEaseMahasiswa/DataSet"

test_dataset_path = "D:/cobaf/AttendEaseMahasiswa/DataTest"

faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
# Fungsi untuk mengunggah dataset manual ke Firebase
# Fungsi untuk mencatat metadata ke Firebase
# Halaman Utama (Opsi Login)
@app.route('/')
def home():
    return render_template('home.html')  

# Login Admin
@app.route('/login_admin', methods=['GET', 'POST'])
def login_admin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Cek login di Firebase
        ref = db.reference('akun')
        accounts = ref.get()

        for user in accounts.values():
            if user['username'] == username and check_password_hash(user['password'], password):
                session['user'] = username
                return redirect('/dashboard')

        return render_template('login_admin.html', message="Username atau Password salah!")

    return render_template('login_admin.html')

# Login Karyawan
@app.route('/login_karyawan', methods=['GET', 'POST'])
def login_karyawan():
    if request.method == 'POST':
        id = request.form['id']
        karyawan_ref = db.reference('employees')
        karyawan_data = karyawan_ref.get()
        for karyawan_id, karyawan_info in karyawan_data.items():
            if karyawan_info['id'] == id:
                session['karyawan'] = {
                    'id': karyawan_info.get('id', ''),
                    'name': karyawan_info.get('name', ''),
                    'jabatan': karyawan_info.get('jabatan', []),
                }
                return redirect('/karyawan_dashboard')

        return render_template('login_karyawan.html', message="ID tidak ditemukan atau tidak terdaftar!")

    return render_template('login_karyawan.html')

from datetime import datetime, timedelta

from datetime import datetime, timedelta

from datetime import datetime, timedelta
import pytz
from datetime import datetime

from datetime import datetime, timedelta
import pytz
from flask import render_template, session, flash, redirect
from firebase_admin import db

@app.route('/karyawan_dashboard')
def karyawan_dashboard():
    if 'karyawan' not in session:
        flash('Silakan login terlebih dahulu', 'warning')
        return redirect('/login_karyawan')

    karyawan = session['karyawan']
    jabatan = karyawan.get('jabatan', '')
    karyawan_id = karyawan.get('id', '')
    jadwal_kerja_data = []
    jadwal_ref = db.reference(f'jadwal_kerja/{jabatan}')
    jadwal_data = jadwal_ref.get()

    # Ambil data absensi hari ini
    today = datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%Y-%m-%d')
    absensi_ref = db.reference('attendance_karyawan')
    absensi_data = absensi_ref.get()

    sudah_absen = {}
    total_absen_per_jadwal = {}

    if absensi_data:
        for jadwal_id, jadwal_val in absensi_data.items():
            if jadwal_id and isinstance(jadwal_val, dict):
                user_absen_data = jadwal_val.get(karyawan_id, {})
                for _, detail in user_absen_data.items():
                    timestamp = detail.get('timestamp', '')
                    if today in timestamp:
                        status = detail.get('status', 'Tidak Hadir')
                        sudah_absen[jadwal_id] = status
                        # Hitung jumlah absensi per jadwal
                        if jadwal_id not in total_absen_per_jadwal:
                            total_absen_per_jadwal[jadwal_id] = 0
                        total_absen_per_jadwal[jadwal_id] += 1

    now = datetime.now()

    if isinstance(jadwal_data, dict):
        for k, v in jadwal_data.items():
            jam_masuk = v.get('jam_masuk', '00:00')
            jam_pulang = v.get('jam_pulang', '23:59')
            toleransi_masuk = 15  # Menit toleransi

            jam_masuk_dt = datetime.strptime(jam_masuk, "%H:%M")
            jam_pulang_dt = datetime.strptime(jam_pulang, "%H:%M")

            waktu_sekarang = now.time()
            batas_awal = jam_masuk_dt.time()
            batas_akhir = (jam_masuk_dt + timedelta(minutes=toleransi_masuk)).time()

            # Status absensi diambil berdasarkan apakah sudah absen atau belum
            status = sudah_absen.get(k, 'Tidak Hadir')
            is_hadir = status == 'Hadir'

            is_dalam_jam_absen = batas_awal <= waktu_sekarang <= batas_akhir
            is_dalam_jam_pulang = waktu_sekarang >= jam_pulang_dt.time()

            # Tentukan status "Hadir" jika sudah absen 2 kali pada jadwal tersebut
            if total_absen_per_jadwal.get(k, 0) >= 2:
                status = 'Hadir'
            else:
                status = 'Tidak Hadir'

            # Perbaikan bagian tombol absen
            show_button_masuk = not is_hadir and is_dalam_jam_absen
            show_button_pulang = is_hadir and is_dalam_jam_pulang

            jadwal_kerja_data.append({
                'id': k,
                'jam_masuk': jam_masuk_dt.time(),  # Convert jam_masuk to time object
                'jam_pulang': jam_pulang_dt.time(),
                'status': status,
                'is_terlambat': not is_dalam_jam_absen and not is_hadir,
                'show_button_masuk': show_button_masuk,  # Menampilkan tombol masuk hanya jika belum absen
                'show_button_pulang': show_button_pulang  # Menampilkan tombol pulang hanya jika sudah absen masuk
            })

    return render_template('karyawan_dashboard.html',
                           karyawan=karyawan,
                           jadwal_kerja_data=jadwal_kerja_data,
                           waktu_sekarang=waktu_sekarang)

# Pengecekan Absensi Hari Ini
def is_hadir_today(jadwal_id, karyawan_id):
    # Ambil data absensi berdasarkan jadwal_id dan karyawan_id dari Firebase
    today = datetime.today().date()
    
    # Loop untuk memeriksa absensi karyawan pada setiap jadwal
    absensi_ref = db.reference('attendance_karyawan')
    absensi_data = absensi_ref.get()
    
    if absensi_data:
        for jadwal_id_key, jadwal_val in absensi_data.items():
            if jadwal_id_key == jadwal_id:
                user_absen_data = jadwal_val.get(karyawan_id, {})
                for _, detail in user_absen_data.items():
                    timestamp = detail.get('timestamp', '')
                    if today.strftime('%Y-%m-%d') in timestamp:
                        if detail.get('status', '') == 'Hadir':
                            return True
    return False



#Register Admin
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if not username or not password:
            return render_template('register.html', message="Username dan Password harus diisi!")

        ref = db.reference('akun')
        akun_data = ref.get()

        if akun_data:
            for user in akun_data.values():
                if user.get('username') == username:
                    return render_template('register.html', message="Username sudah digunakan, silakan pilih username lain!")

        hashed_password = generate_password_hash(password)

        try:
            ref.push({
                'username': username,
                'password': hashed_password
            })
            flash('Registrasi berhasil! Silakan login.', 'success')  # Tambahkan flash message
            return redirect('/login_admin')
        except Exception as e:
            return render_template('register.html', message=f"Terjadi kesalahan: {str(e)}")

    return render_template('register.html')

# Halaman Dashboard Admin
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/')
    return render_template('dashboard.html')

from firebase_admin import db

@app.route('/kasbon/add', methods=['POST'])
def add_kasbon():
    if 'user' not in session:
        return redirect('/')

    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'Data tidak ditemukan.'})

    try:
        employee_id = data['employeeId']
        name = data['name']
        jabatan = data['jabatan']
        kasbon_to_add = int(data['kasbon'])

        # Mendapatkan data kasbon yang sudah ada
        kasbon_ref = db.reference(f'kasbon/{employee_id}')
        current_kasbon_data = kasbon_ref.get()

        if current_kasbon_data:
            # Jika kasbon sudah ada, tambahkan jumlah kasbon yang baru
            current_kasbon = current_kasbon_data.get('kasbon', 0)
            updated_kasbon = current_kasbon + kasbon_to_add
        else:
            # Jika kasbon belum ada, set kasbon pertama kali
            updated_kasbon = kasbon_to_add

        # Update data kasbon yang telah diperbarui
        kasbon_ref.update({
            'name': name,
            'jabatan': jabatan,
            'kasbon': updated_kasbon
        })

        return jsonify({'status': 'success', 'message': 'Kasbon berhasil disimpan.'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Gagal menyimpan kasbon: {str(e)}'})


# Endpoint untuk mendapatkan data kasbon
@app.route('/kasbon/<employee_id>', methods=['GET'])
def get_kasbon(employee_id):
    if 'user' not in session:
        return redirect('/')

    kasbon_ref = db.reference(f'kasbon/{employee_id}')
    kasbon_data = kasbon_ref.get()

    if kasbon_data:
        return jsonify(kasbon_data)
    else:
        return jsonify({'status': 'error', 'message': 'Data kasbon tidak ditemukan.'})


import time


# Generator untuk streaming dan face-recog
# Generator untuk streaming dan face recognition absensi karyawan
import cv2
import os
import numpy as np
from datetime import datetime
from firebase_admin import db, storage
from flask import Response, jsonify, redirect, render_template, request, session

# Fungsi untuk menangani absensi dan pengambilan gambar wajah karyawan
# Fungsi untuk menangani absensi dan pengambilan gambar wajah karyawan
def gen(user_id, jadwal_id, jam_masuk, jam_pulang):
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("[ERROR] Kamera tidak dapat dibuka.")
        return

    attendance_logged = False
    capture_dir = 'captures'
    os.makedirs(capture_dir, exist_ok=True)

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (224, 224))
                face_img = np.expand_dims(face_img, axis=0) / 255.0

                pred = model.predict(face_img)
                confidence = float(np.max(pred[0]))
                detected_idx = str(np.argmax(pred[0]) + 1)

                detected_user_id = f"KRY-{detected_idx.zfill(2)}"
                detected_label = labels.get(detected_idx, 'Unknown')

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{detected_user_id} ({confidence*100:.1f}%)", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                print(f"[INFO] Expected: {user_id} | Detected: {detected_user_id} | Confidence: {confidence:.2f}")

                if detected_user_id == user_id and confidence >= 0.75 and not attendance_logged:
                    timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
                    filename = f"att_{user_id}_{jadwal_id}_{timestamp}.jpg"
                    filepath = os.path.join(capture_dir, filename)

                    cv2.imwrite(filepath, frame)
                    print(f"[DEBUG] Gambar berhasil disimpan di {filepath}")

                    bucket = storage.bucket()
                    blob = bucket.blob(f'attendance_images/{filename}')
                    blob.upload_from_filename(filepath)
                    blob.make_public()
                    image_url = blob.public_url

                    # ✅ Ambil data nama dan jabatan dari Firebase
                    employee_ref = db.reference(f"employees/{user_id}")
                    employee_data = employee_ref.get()

                    if not employee_data:
                        print(f"[ERROR] Data karyawan dengan ID {user_id} tidak ditemukan.")
                        continue

                    employee_name = employee_data.get('name', 'Unknown')
                    employee_jabatan = employee_data.get('jabatan', 'Unknown')

                    # ✅ Simpan absensi ke Firebase
                    ref = db.reference(f"attendance_karyawan/{jadwal_id}/{user_id}")
                    data = {
                        'id_jadwal': jadwal_id,
                        'jam_masuk': jam_masuk,
                        'jam_pulang': jam_pulang,
                        'timestamp': timestamp,
                        'image_url': image_url,
                        'status': 'Hadir',
                        'name': employee_name,
                        'jabatan': employee_jabatan
                    }
                    ref.push(data)
                    print("[SUCCESS] Absensi tercatat di Firebase:", data)
                    attendance_logged = True

            ret, buf = cv2.imencode('.jpg', frame)
            frame_bytes = buf.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            if attendance_logged:
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


# Route untuk streaming video
@app.route('/video_feed/<user_id>/<jadwal_id>/<jam_masuk>/<jam_pulang>')
def video_feed(user_id, jadwal_id, jam_masuk, jam_pulang):
    return Response(gen(user_id, jadwal_id, jam_masuk, jam_pulang),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route untuk halaman absen karyawan
# Route untuk halaman absen karyawan
@app.route('/absen', methods=['GET', 'POST'])
def absen():
    if 'karyawan' not in session:
        return redirect('/login_karyawan')

    karyawan = session['karyawan']
    user_id = karyawan.get('id')
    jadwal_id = request.args.get('jadwal_id')
    jam_masuk = request.args.get('jam_masuk')
    jam_pulang = request.args.get('jam_pulang')

    if not all([jadwal_id, jam_masuk, jam_pulang]):
        return "Parameter jadwal tidak lengkap", 400

    if request.method == 'POST':
        # Lakukan pengecekan wajah dan absensi
        attendance_logged = False
        # Simulasi proses absensi di kamera
        timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        filename = f"att_{user_id}_{jadwal_id}_{timestamp}.jpg"
        filepath = os.path.join(capture_dir, filename)

        # Jika wajah terdeteksi dan absensi tercatat
        if attendance_logged:
            # Update status absensi karyawan menjadi "Hadir" setelah berhasil absen
            attendance_ref = db.reference(f"attendance_karyawan/{jadwal_id}/{user_id}")
            attendance_ref.update({
                'status': 'Hadir',
                'jam_masuk': jam_masuk,
                'timestamp': timestamp
            })

            flash("Absen masuk berhasil!", "success")
            return redirect(url_for('karyawan_dashboard'))  # Kembali ke halaman dashboard setelah absen

        return jsonify({'status': 'error', 'message': 'Terjadi kesalahan saat menyimpan absensi.'}), 500

    # Jika metode GET, tampilkan halaman absensi
    return render_template('absen.html', karyawan=karyawan, jadwal_id=jadwal_id, jam_masuk=jam_masuk, jam_pulang=jam_pulang)

# Route untuk memeriksa status absensi karyawan
@app.route('/check_absen_status_karyawan', methods=['GET'])
def check_absen_status_karyawan():
    user_id = request.args.get('user_id')
    jadwal_id = request.args.get('jadwal_id')

    if not user_id or not jadwal_id:
        print("[ERROR] Parameter tidak lengkap. Pastikan 'user_id' dan 'jadwal_id' disertakan.")
        return jsonify({"status": "error", "message": "Parameter tidak lengkap"}), 400

    print(f"[DEBUG] Checking attendance for user_id={user_id}, jadwal_id={jadwal_id}")

    try:
        # Referensi ke database karyawan
        attendance_ref = db.reference(f"attendance_karyawan/{jadwal_id}/{user_id}")

        data = attendance_ref.get()

        if data:
            print("[DEBUG] Attendance Data Found:", data)
            return jsonify({"status": "success", "data": data})
        else:
            print("[DEBUG] Attendance Data Not Found")
            return jsonify({"status": "pending", "message": "Data absensi tidak ditemukan"}), 404
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan saat memeriksa status absensi: {e}")
        return jsonify({"status": "error", "message": "Terjadi kesalahan server"}), 500


def generate_new_employee_id():
    employees_ref = db.reference('employees')
    all_employees = employees_ref.get() or {}

    nums = []
    for emp in all_employees.values():
        emp_id = emp.get('id', '')
        if emp_id.startswith('KRY-') and emp_id.split('-')[-1].isdigit():
            nums.append(int(emp_id.split('-')[-1]))

    new_num = max(nums) + 1 if nums else 1
    return f'KRY-{new_num:02d}'


@app.route('/api/get-last-id', methods=['GET'])
def get_last_id():
    try:
        new_id = generate_new_employee_id()
        return jsonify({'employee_id': new_id})   # KEY HARUS employee_id
    except Exception as e:
        # log error lengkap ke console
        app.logger.exception(e)
        return jsonify({'error': str(e)}), 500

@app.route('/dataset', methods=['GET', 'POST'])
def dataset():
    if request.method == 'GET':
        return render_template('dataset.html')

    elif request.method == 'POST':
        try:
            # Ambil data dari form
            name = request.form.get('name')
            jabatan = request.form.get('jabatan')
            employee_id = request.form.get('employee_id')

            if not all([name, jabatan, employee_id]):
                return jsonify({'status': 'error', 'message': 'Nama, Jabatan, dan ID Karyawan harus diisi!'}), 400

            # Cek apakah employee_id sudah ada
            employee_ref = db.reference(f'employees/{employee_id}')
            if employee_ref.get() is not None:
                return jsonify({
                    'status': 'error',
                    'message': f'ID {employee_id} sudah terdaftar. Gunakan ID lain!'
                }), 400

            # Siapkan folder lokal
            folder_name = f"{employee_id}-{name.replace(' ', '_')}"
            folder_path = os.path.join(dataset_path, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            # Ambil file gambar dari request
            images = [request.form.get(key) for key in request.form if key.startswith('image_')]
            if not images:
                return jsonify({'status': 'error', 'message': 'Tidak ada gambar yang diterima!'}), 400

            def process_and_crop_faces(image, file_name_prefix, save_folder, user_id, user_name, start_count=0, padding=0.2):
                img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                image = cv2.medianBlur(image, 5)

                faces = face_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
                count = start_count

                for (x, y, w, h) in faces:
                    x_pad, y_pad = int(padding * w), int(padding * h)
                    x_start, y_start = max(0, x - x_pad), max(0, y - y_pad)
                    x_end, y_end = min(image.shape[1], x + w + x_pad), min(image.shape[0], y + h + y_pad)

                    cropped_face = image[y_start:y_end, x_start:x_end]
                    resized_face = cv2.resize(cropped_face, (224, 224), interpolation=cv2.INTER_AREA)

                    file_name = f"{file_name_prefix}.{count}.jpg"
                    local_path = os.path.join(save_folder, file_name)
                    cv2.imwrite(local_path, resized_face)

                    blob = bucket.blob(f'images/{user_id}_{user_name}/{file_name}')
                    blob.upload_from_filename(local_path)
                    blob.make_public()
                    count += 1

                return count

            total_faces_saved = 0
            for idx, image_data in enumerate(images):
                try:
                    image_data = image_data.split(",")[1]
                    img_array = np.frombuffer(base64.b64decode(image_data), np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    if img is None:
                        return jsonify({'status': 'error', 'message': f'Gambar ke-{idx + 1} tidak valid!'}), 400

                    total_faces_saved = process_and_crop_faces(
                        image=img,
                        file_name_prefix=employee_id,
                        save_folder=folder_path,
                        user_id=employee_id,
                        user_name=name,
                        start_count=total_faces_saved
                    )
                except Exception as e:
                    print(f"Kesalahan pada gambar ke-{idx + 1}: {str(e)}")

            if total_faces_saved == 0:
                return jsonify({'status': 'error', 'message': 'Tidak ada wajah yang berhasil disimpan!'}), 400

            # Simpan metadata ke Firebase
            images_count = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            employee_ref.set({
                'id': employee_id,
                'name': name,
                'id_karyawan': employee_id.split('-')[-1],
                'jabatan': jabatan,
                'images_count': images_count,
                'timestamp': datetime.now().isoformat()
            })

            return jsonify({
                'status': 'success',
                'message': f'Dataset berhasil disimpan. Total wajah: {total_faces_saved}',
                'employee_id': employee_id
            })

        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Terjadi kesalahan: {str(e)}'}), 500

# Fungsi untuk mengunggah file dan mencatat metadata
def upload_dataset_to_firebase():
    try:
        # Firebase references
        bucket = storage.bucket()
        employees_ref = db.reference('employees')

        print(f"Memproses folder dataset: {dataset_path}")

        for employee_folder in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, employee_folder)

            # Abaikan jika bukan folder
            if not os.path.isdir(folder_path):
                continue

            print(f"Memproses folder: {employee_folder}")

            # Ekstrak ID Karyawan dari folder
            if "Karyawan-" in employee_folder:
                id_karyawan = employee_folder.split("Karyawan-")[1]
            else:
                print(f"Folder {employee_folder} tidak memiliki format 'Karyawan-{id_karyawan}'. Abaikan.")
                continue

            # Hitung jumlah file gambar
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            images_count = len(image_files)

            # Unggah setiap file gambar ke Firebase Storage
            for image_file in image_files:
                local_file_path = os.path.join(folder_path, image_file)
                cloud_file_path = f"datasets/{employee_folder}/{image_file}"

                # Upload file ke Firebase Storage
                blob = bucket.blob(cloud_file_path)
                blob.upload_from_filename(local_file_path)
                blob.make_public()  # Buat file bisa diakses publik
                print(f"File diunggah: {local_file_path} -> {cloud_file_path}")

            # Simpan metadata ke Realtime Database
            metadata = {
                'id': employee_folder,
                'id_karyawan': id_karyawan,
                'name': employee_folder.split('', 1)[1] if '' in employee_folder else "Unknown",
                'jabatan': "Manager",
                'images_count': images_count,
                'timestamp': datetime.now().isoformat()
            }
            employees_ref.child(employee_folder).set(metadata)

            print(f"Metadata untuk {employee_folder} berhasil disimpan: {metadata}")

    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

# Jalankan fungsi
upload_dataset_to_firebase()



@app.route("/attendance", methods=["GET", "POST"])
def attendance():
    snapshot = db.reference('attendance_karyawan').get()
    attendance_list = []
    
    if snapshot:
        grouped_data = defaultdict(list)

        # Kelompokkan data per karyawan per tanggal
        for id_jadwal, karyawan_data in snapshot.items():
            for id_karyawan, records in karyawan_data.items():
                for record_id, details in records.items():
                    try:
                        dt_object = datetime.strptime(details.get("timestamp", ""), "%Y-%m-%dT%H-%M-%S")
                        tanggal = dt_object.strftime("%Y-%m-%d")
                    except:
                        continue  # Lewati jika format timestamp tidak valid

                    grouped_data[(id_karyawan, tanggal)].append({
                        "jam_masuk": details.get("jam_masuk", ""),
                        "jam_pulang": details.get("jam_pulang", ""),
                        "image_url": details.get("image_url", ""),
                        "timestamp": dt_object,  # Simpan sebagai datetime object
                        "name": details.get("name", ""),
                    })

        # Proses status kehadiran
        for (id_karyawan, tanggal), entries in grouped_data.items():
            # Urutkan berdasarkan timestamp
            entries.sort(key=lambda x: x["timestamp"])
            
            # Tentukan status berdasarkan jumlah presensi
            if len(entries) >= 2:
                # Ambil record pertama sebagai masuk dan terakhir sebagai pulang
                masuk_record = entries[0]
                pulang_record = entries[-1]
                
                status = "Hadir"
                jam_masuk = masuk_record["jam_masuk"] or "-"
                jam_pulang = pulang_record["jam_pulang"] or "-"
                bukti_masuk = masuk_record["image_url"]
                bukti_pulang = pulang_record["image_url"]
            else:
                status = "Tidak Hadir"
                jam_masuk = entries[0]["jam_masuk"] if entries else "-"
                jam_pulang = "-"
                bukti_masuk = entries[0]["image_url"] if entries else ""
                bukti_pulang = ""

            attendance_list.append({
                "id_karyawan": id_karyawan,
                "nama_karyawan": entries[0]["name"] if entries else "-",
                "tanggal": tanggal,
                "jam_masuk": jam_masuk,
                "jam_pulang": jam_pulang,
                "status": status,
                "bukti_masuk": bukti_masuk,
                "bukti_pulang": bukti_pulang,
                "jumlah_presensi": len(entries)  # Untuk debugging
            })

    # Filtering
    if request.method == "POST":
        id_karyawan_filter = request.form.get("id_karyawan")
        tanggal_filter = request.form.get("tanggal")
        nama_filter = request.form.get("nama_karyawan")

        if id_karyawan_filter:
            attendance_list = [a for a in attendance_list if id_karyawan_filter.lower() in a['id_karyawan'].lower()]
        if tanggal_filter:
            attendance_list = [a for a in attendance_list if a['tanggal'] == tanggal_filter]
        if nama_filter:
            attendance_list = [a for a in attendance_list if nama_filter.lower() in a['nama_karyawan'].lower()]

    return render_template("attendance.html", attendance_list=attendance_list)


@app.route('/admin/penggajian', methods=['GET', 'POST'])
def admin_penggajian():
    if 'user' not in session:
        return redirect('/login_admin')

    # Ambil gaji default dari Firebase
    default_gaji_ref = db.reference('default_gaji/default')
    gaji_default_data = default_gaji_ref.get()

    if not gaji_default_data:
        default_gaji_ref.set({'gaji': 10000})
        gaji_default = 10000
    else:
        gaji_default = gaji_default_data.get('gaji', 10000)

    if request.method == 'POST' and 'gaji_default' in request.form:
        try:
            new_gaji = int(request.form['gaji_default'])
            default_gaji_ref.update({'gaji': new_gaji})
            flash('Gaji default berhasil diupdate', 'success')
        except ValueError:
            flash('Nilai gaji harus berupa angka', 'error')
        return redirect('/admin/penggajian')

    nama_karyawan = request.form.get('nama_karyawan', '')
    tanggal_mulai = request.form.get('tanggal_mulai', '')
    tanggal_selesai = request.form.get('tanggal_selesai', '')
    attendance_list = []

    attendance_ref = db.reference('attendance_karyawan')
    attendance_data = attendance_ref.get() or {}

    employees_ref = db.reference('employees')
    employees_data = employees_ref.get() or {}

    # Daftar nama karyawan
    nama_karyawan_set = set()
    for emp_id, emp_info in employees_data.items():
        nama = emp_info.get('nama', '-')
        if nama != "-":
            nama_karyawan_set.add(nama)

    for jadwal_id, karyawan_data in attendance_data.items():
        for karyawan_id, records in karyawan_data.items():
            for record_id, detail in records.items():
                nama = detail.get("name", "-")
                if nama and nama != "-":
                    nama_karyawan_set.add(nama)

    nama_karyawan_list = sorted(list(nama_karyawan_set))

    def str_to_date(date_str):
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except:
            return None

    tanggal_mulai_obj = str_to_date(tanggal_mulai) if tanggal_mulai else None
    tanggal_selesai_obj = str_to_date(tanggal_selesai) if tanggal_selesai else None

    grouped_attendance = defaultdict(list)

    for jadwal_id, karyawan_data in attendance_data.items():
        for karyawan_id, records in karyawan_data.items():
            for record_id, detail in records.items():
                nama = detail.get("name", "-")
                if nama_karyawan and nama != nama_karyawan:
                    continue

                timestamp_str = detail.get("timestamp", "")
                timestamp = None
                if isinstance(timestamp_str, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp_str)
                else:
                    try:
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
                    except ValueError:
                        try:
                            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S")
                        except ValueError:
                            continue

                if tanggal_mulai_obj and timestamp.date() < tanggal_mulai_obj.date():
                    continue
                if tanggal_selesai_obj and timestamp.date() > tanggal_selesai_obj.date():
                    continue

                tanggal_key = timestamp.date()
                grouped_attendance[(karyawan_id, nama, tanggal_key)].append({
                    'timestamp': timestamp,
                    'status': detail.get('status', 'Hadir'),
                    'karyawan_id': karyawan_id  # Tambahkan ini untuk referensi
                })

    for (karyawan_id, nama, tanggal_key), records in grouped_attendance.items():
        records.sort(key=lambda r: r['timestamp'])
        jumlah_presensi = len(records)
        
        # Logika status hadir (minimal 2 presensi)
        status = 'Hadir' if jumlah_presensi >= 2 else 'Tidak Hadir'
        gaji = gaji_default if status == 'Hadir' else 0
        
        minggu_ke = records[0]['timestamp'].isocalendar()[1] if records else '-'
        tanggal_ditampilkan = tanggal_key.strftime('%d %B %Y') if records else '-'
        
        status_gaji = 'belum diambil'

        # Cek status penggajian di Firebase
        penggajian_ref = db.reference(f"penggajian/{karyawan_id}")
        penggajian_data = penggajian_ref.get()

        if penggajian_data:
            for penggajian_item in penggajian_data.values():
                detail_list = penggajian_item.get('detail', [])
                for detail in detail_list:
                    tanggal_penggajian = detail.get('tanggal', '')
                    if tanggal_penggajian == tanggal_ditampilkan:
                        if detail.get('status') == 'sudah diambil':
                            status_gaji = 'sudah diambil'
                            break
                if status_gaji == 'sudah diambil':
                    break

        attendance_list.append({
            "id_karyawan": karyawan_id,
            "nama": nama,
            "status": status,
            "tanggal": tanggal_ditampilkan,
            "minggu_ke": minggu_ke,
            "gaji": gaji,
            "status_gaji": status_gaji,
            "jumlah_presensi": jumlah_presensi  # Untuk debugging
        })



    attendance_list = sorted(attendance_list, key=lambda x: (str(x['minggu_ke']), x['nama']))

    return render_template('penggajian.html',
                         attendance_list=attendance_list,
                         nama_karyawan_list=nama_karyawan_list,
                         nama_karyawan=nama_karyawan,
                         gaji_default=gaji_default)
    
@app.route("/admin/ambil_kasbon", methods=["POST"])
def ambil_kasbon():
    data = request.get_json()
    id_karyawan = data.get("id_karyawan")

    if not id_karyawan:
        return jsonify({"success": False, "message": "ID karyawan tidak diberikan."})

    try:
        # Akses kasbon dari Firebase berdasarkan ID
        ref_kasbon = db.reference(f"kasbon/{id_karyawan}")
        data_kasbon = ref_kasbon.get()

        if not data_kasbon:
            return jsonify({"success": False, "message": "Data kasbon tidak ditemukan."})

        jumlah_kasbon = int(data_kasbon.get("kasbon", 0))
        nama = data_kasbon.get("name", "-")
        jabatan = data_kasbon.get("jabatan", "-")

        return jsonify({
            "success": True,
            "id_karyawan": id_karyawan,
            "kasbon": jumlah_kasbon,
            "name": nama,
            "jabatan": jabatan
        })

    except Exception as e:
        print("Error saat ambil kasbon:", e)
        return jsonify({"success": False, "message": "Terjadi kesalahan saat ambil kasbon."})

@app.route('/admin/proses_ambil_gaji', methods=['POST'])
def proses_ambil_gaji():
    if 'user' not in session:
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401

    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'Data tidak ditemukan.'}), 400

        # Data utama
        employee_id = data.get('employeeId')
        nama = data.get('nama', '')
        detail = data.get('detail', [])
        total_gaji = data.get('total_gaji', 0)
        total_kasbon = data.get('total_kasbon', 0)
        sisa_gaji = data.get('sisa_gaji', 0)
        
        # Waktu transaksi
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tanggal = data.get('tanggal', datetime.now().strftime('%Y-%m-%d'))
        
        # Referensi Firebase
        penggajian_ref = db.reference(f'penggajian/{employee_id}/{timestamp}')
        histori_ref = db.reference(f'histori_penggajian/{employee_id}')
        
        # Data yang akan disimpan
        penggajian_data = {
            'employee_id': employee_id,
            'nama': nama,
            'detail': detail,
            'total_gaji': total_gaji,
            'total_kasbon': total_kasbon,
            'sisa_gaji': sisa_gaji,
            'status': 'sudah diambil',
            'tanggal': tanggal,
            'timestamp': datetime.now().isoformat()
        }
        
        # Simpan ke dua lokasi berbeda
        penggajian_ref.set(penggajian_data)
        histori_ref.push().set(penggajian_data)
        
        # Update status di data attendance
        for item in detail:
            if 'id_jadwal' in item:
                attendance_ref = db.reference(
                    f"attendance_karyawan/{item['id_jadwal']}/{employee_id}"
                )
                attendance_data = attendance_ref.get()
                if attendance_data:
                    for key, val in attendance_data.items():
                        if isinstance(val, dict):
                            attendance_ref.child(key).update({
                                'status_gaji': 'sudah diambil',
                                'tanggal_penggajian': tanggal
                            })
        
        # Jika ada kasbon, update kasbon jadi 0
        if total_kasbon > 0:
            kasbon_ref = db.reference(f'kasbon/{employee_id}/kasbon')
            kasbon_ref.set(0)
        
        return jsonify({
            'status': 'success',
            'message': 'Data penggajian berhasil disimpan dan kasbon telah dilunasi',
            'data': {
                'employee_id': employee_id,
                'nama': nama,
                'total_gaji': data['total_gaji'],
                'total_kasbon': data['total_kasbon'],
                'sisa_gaji': data['sisa_gaji'],
                'status': 'sudah diambil',
                'tanggal': datetime.now().strftime('%Y-%m-%d'),
                'kasbon_updated': True  # Flag untuk menunjukkan kasbon sudah diupdate
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/proses_gaji', methods=['POST'])
def proses_gaji():
    try:
        # Validasi request
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Data harus JSON'}), 400
            
        data = request.get_json()
        
        # Validasi field wajib
        required_fields = ['id_karyawan', 'total_gaji', 'kasbon', 'sisa_gaji', 'tanggal']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Field {field} harus diisi'}), 400
        
        id_karyawan = str(data['id_karyawan']).strip()
        if not id_karyawan:
            return jsonify({'success': False, 'error': 'ID Karyawan tidak valid'}), 400
            
        try:
            total_gaji = int(data['total_gaji'])
            kasbon = int(data['kasbon'])
            sisa_gaji = int(data['sisa_gaji'])
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': 'Nilai gaji/kasbon harus angka'}), 400

        tanggal_gaji = data['tanggal']
        if not isinstance(tanggal_gaji, str) or not tanggal_gaji.strip():
            return jsonify({'success': False, 'error': 'Tanggal tidak valid'}), 400

        # Proses penyimpanan
        now = datetime.now()
        timestamp_str = now.strftime('%d %B %Y, %H:%M')
        
        # Simpan ke detail_gaji
        detail_ref = db.reference(f'detail_gaji/{id_karyawan}')
        new_detail_ref = detail_ref.push()
        new_detail_ref.set({
            'status': 'Sudah diambil',
            'gaji': total_gaji,
            'tanggal': tanggal_gaji,
            'timestamp': now.timestamp(),
            'processed_at': timestamp_str
        })
        
        # Simpan ke gaji
        gaji_ref = db.reference(f'gaji/{id_karyawan}')
        new_gaji_ref = gaji_ref.push()
        new_gaji_ref.set({
            'tanggal_pengambilan': timestamp_str,
            'total_gaji': total_gaji,
            'kasbon': kasbon,
            'sisa_gaji': sisa_gaji,
            'status': 'Completed',
            'timestamp': now.timestamp(),
            'detail_gaji_ref': new_detail_ref.key
        })
        
        return jsonify({
            'success': True,
            'message': 'Gaji berhasil diproses',
            'data': {
                'id_karyawan': id_karyawan,
                'total_gaji': total_gaji,
                'kasbon': kasbon,
                'sisa_gaji': sisa_gaji,
                'tanggal': tanggal_gaji
            }
        })
        
    except Exception as e:
        print(f"ERROR in proses_gaji: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Terjadi kesalahan sistem',
            'message': str(e)
        }), 500

# @app.route('/students/edit/<student_id>', methods=['POST'])
# def edit_student(student_id):
#     """
#     Endpoint untuk mengedit data mahasiswa berdasarkan student_id.
#     """
#     try:
#         data = request.json  # Data dikirim dalam format JSON dari frontend
#         student_ref = db.reference(f'students/{student_id}')
        
#         # Perbarui data mahasiswa di Firebase
#         updated_data = {
#             'semester': data.get('semester', ''),
#             'golongan': data.get('golongan', '')
#         }
#         student_ref.update(updated_data)

#         return jsonify({'status': 'success', 'message': f'Data mahasiswa dengan ID {student_id} berhasil diperbarui.'})
#     except Exception as e:
#         print(f"Error saat memperbarui data mahasiswa: {str(e)}")
#         return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/employees/edit/<employee_id>', methods=['POST'])
def edit_employee(employee_id):
    """
    Endpoint untuk mengedit data karyawan berdasarkan employee_id.
    """
    try:
        data = request.json  # Data dikirim dalam format JSON dari frontend
        employee_ref = db.reference(f'employees/{employee_id}')
        
        # Perbarui data karyawan di Firebase
        updated_data = {
            'name': data.get('name', ''),
            'jabatan': data.get('jabatan', '')
        }
        employee_ref.update(updated_data)

        return jsonify({'status': 'success', 'message': f'Data karyawan dengan ID {employee_id} berhasil diperbarui.'})
    except Exception as e:
        print(f"Error saat memperbarui data karyawan: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/employees/delete/<employee_id>', methods=['DELETE'])
def delete_employee(employee_id):
    try:
        employee_ref = db.reference(f'employees/{employee_id}')
        if employee_ref.get():
            employee_ref.delete()
            return jsonify({'status': 'success', 'message': f'Data karyawan dengan ID {employee_id} berhasil dihapus.'})
        else:
            return jsonify({'status': 'error', 'message': f'Data dengan ID {employee_id} tidak ditemukan.'}), 404
    except Exception as e:
        print(f"Error saat menghapus data: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/employees', methods=['GET'])
def get_employees():
    try:
        employees_data = []
        employees_ref = db.reference('employees')
        employees = employees_ref.get()

        print("Data dari Firebase:", employees)

        if employees:
            for emp_id, emp_info in employees.items():
                employees_data.append({
                    'id': emp_id,
                    'name': emp_info.get('name', 'Unknown'),
                    'jabatan': emp_info.get('jabatan', 'Unknown'),
                    'images_count': emp_info.get('images_count', 0),
                    'edit_url': f'/employees/edit/{emp_id}',
                    'delete_url': f'/employees/delete/{emp_id}'
                })

        return jsonify({'status': 'success', 'data': employees_data})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/admin/jadwal_mata_kuliah/delete/<golongan>/<jadwal_id>', methods=['POST'])
def delete_jadwal(golongan, jadwal_id):
    """
    Route untuk menghapus jadwal mata kuliah berdasarkan golongan dan ID jadwal.
    """
    if 'user' not in session:
        return redirect('/')

    try:
        # Hapus jadwal dari Firebase berdasarkan ID
        jadwal_ref = db.reference(f'jadwal_mata_kuliah/{golongan}/{jadwal_id}')
        jadwal_ref.delete()
        print(f"[INFO] Jadwal {jadwal_id} berhasil dihapus dari golongan {golongan}")
    except Exception as e:
        print(f"[ERROR] Gagal menghapus jadwal {jadwal_id} dari golongan {golongan}: {e}")

    return redirect('/admin/jadwal_mata_kuliah')
def generate_new_schedule_id():
    ref = db.reference('jadwal_kerja')
    all_data = ref.get() or {}

    nums = []
    for jabatan in all_data:
        for key in all_data[jabatan]:
            if key.startswith("JD-") and key[3:].isdigit():
                nums.append(int(key[3:]))

    new_num = max(nums) + 1 if nums else 1
    return f'JD-{new_num:03d}'

@app.route('/admin/jadwal_kerja', methods=['GET', 'POST'])
def admin_jadwal_kerja():
    ref = db.reference('jadwal_kerja')

    if request.method == 'POST':
        action = request.form['action']

        if action == 'add':
            jabatan = request.form['jabatan']
            new_id = generate_new_schedule_id()
            data = {
                'id_jadwal': new_id,
                'jabatan': jabatan,
                'jam_masuk': request.form['jam_masuk'],
                'jam_pulang': request.form['jam_pulang'],
                'toleransi_keterlambatan': int(request.form['toleransi']),
            }
            ref.child(jabatan).child(new_id).set(data)

        elif action == 'delete':
            jabatan = request.form['jabatan']
            del_id = request.form['del_id']
            ref.child(jabatan).child(del_id).delete()

        elif action == 'prepare_edit':
            jabatan = request.form['jabatan']
            edit_id = request.form['edit_id']
            edit_data = ref.child(jabatan).child(edit_id).get()
            all_data = ref.get() or {}
            jadwal_list = []
            for jab, jads in all_data.items():
                for item in jads.values():
                    jadwal_list.append(item)
            return render_template('admin_jadwal_kerja.html',
                                   data_list=jadwal_list,
                                   edit_data=edit_data,
                                   edit_jabatan=jabatan)

        elif action == 'update':
            jabatan = request.form['jabatan']
            id_jadwal = request.form['id_jadwal']
            ref.child(jabatan).child(id_jadwal).update({
                'jabatan': jabatan,
                'jam_masuk': request.form['jam_masuk'],
                'jam_pulang': request.form['jam_pulang'],
                'toleransi_keterlambatan': int(request.form['toleransi']),
            })

        return redirect('/admin/jadwal_kerja')

    # GET
    all_data = ref.get() or {}
    jadwal_list = []
    for jab, jads in all_data.items():
        for item in jads.values():
            jadwal_list.append(item)

    return render_template('admin_jadwal_kerja.html',
                           data_list=jadwal_list,
                           edit_data=None,
                           edit_jabatan=None,
                           new_id_jadwal=generate_new_schedule_id())
# Admin Mengatur Jadwal Absensi
@app.route('/set_absensi', methods=['GET', 'POST'])
def set_absensi():
    if 'user' not in session:
        return redirect('/')

    if request.method == 'POST':
        mata_kuliah = request.form['mata_kuliah']
        tanggal = request.form['tanggal']
        start_time = request.form['start_time']

        try:
            # Simpan waktu awal dalam format timestamp
            start_timestamp = datetime.strptime(f"{tanggal} {start_time}", "%Y-%m-%d %H:%M").timestamp()

            # Simpan jadwal absensi ke Firebase
            attendance_ref = db.reference(f'attendance/{mata_kuliah}/{tanggal}')
            attendance_ref.set({
                'start_time': start_timestamp,
                'students': {}
            })

            return render_template('set_absensi.html', message="Jadwal absensi berhasil disimpan!")
        except Exception as e:
            return render_template('set_absensi.html', message=f"Terjadi kesalahan: {str(e)}")

    return render_template('set_absensi.html', message=None)
    
from flask import Flask, render_template, request, redirect, session
from datetime import datetime, timedelta, date
from calendar import monthrange



@app.route('/rekap_absensi', methods=['GET'])
def rekap_absensi():
    if 'karyawan' not in session:
        return redirect('/login_karyawan')

    karyawan = session['karyawan']
    id_karyawan = karyawan['id']

    attendance_ref = db.reference("attendance_karyawan")
    semua_absensi = attendance_ref.get() or {}

    semua_tanggal_hadir = set()
    data_hadir_detail = defaultdict(list)
    bulan_terdata = set()

    # Kumpulkan data absensi per tanggal dan catat bulan-tahun yg ada data
    for id_jadwal, karyawan_data in semua_absensi.items():
        record_karyawan = karyawan_data.get(id_karyawan, {})
        for record_id, detail in record_karyawan.items():
            raw_ts = detail.get('timestamp', '')
            try:
                tanggal = datetime.strptime(raw_ts, "%Y-%m-%dT%H-%M-%S").date()
                semua_tanggal_hadir.add(tanggal)
                
                # Tambahkan id_jadwal ke data detail
                detail_with_id = detail.copy()
                detail_with_id['id_jadwal'] = id_jadwal
                
                data_hadir_detail[tanggal].append(detail_with_id)
                bulan_terdata.add((tanggal.year, tanggal.month))
            except ValueError:
                continue

    # Ambil filter bulan dari parameter GET
    bulan_filter = request.args.get('month', type=int)
    tahun_filter = datetime.today().year
    if not bulan_filter:
        bulan_filter = datetime.today().month

    hasil_rekap = []
    if (tahun_filter, bulan_filter) in bulan_terdata:
        jumlah_hari = monthrange(tahun_filter, bulan_filter)[1]
        today = datetime.today().date()
        if tahun_filter == today.year and bulan_filter == today.month:
            jumlah_hari = today.day

        tanggal_semua = [date(tahun_filter, bulan_filter, day) for day in range(1, jumlah_hari + 1)]

        for tanggal in tanggal_semua:
            if tanggal in semua_tanggal_hadir:
                records = data_hadir_detail[tanggal]
                records.sort(key=lambda x: x['timestamp'])

                if len(records) >= 2:
                    status = 'Hadir'
                    jam_masuk = records[0].get('jam_masuk', '-')
                    jam_pulang = records[-1].get('jam_pulang', '-')
                    image_url_masuk = records[0].get('image_url', '')
                    image_url_pulang = records[-1].get('image_url', '')
                    id_jadwal = records[0].get('id_jadwal', '-')
                else:
                    status = 'Tidak Hadir'
                    jam_masuk = records[0].get('jam_masuk', '-') if records else '-'
                    jam_pulang = '-'
                    image_url_masuk = records[0].get('image_url', '') if records else ''
                    image_url_pulang = ''
                    id_jadwal = records[0].get('id_jadwal', '-') if records else '-'
            else:
                status = 'Tidak Hadir'
                jam_masuk = '-'
                jam_pulang = '-'
                image_url_masuk = ''
                image_url_pulang = ''
                id_jadwal = '-'

            hasil_rekap.append({
                'id_jadwal': id_jadwal,
                'tanggal': tanggal.strftime('%Y-%m-%d'),
                'jam_masuk': jam_masuk,
                'jam_pulang': jam_pulang,
                'status': status,
                'image_url_masuk': image_url_masuk,
                'image_url_pulang': image_url_pulang
            })

    # Pagination
    page = request.args.get('page', default=1, type=int)
    per_page = 7
    total = len(hasil_rekap)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_absensi = hasil_rekap[start:end]
    total_pages = (total + per_page - 1) // per_page

    return render_template(
        "rekap_absensi.html",
        absensi_list=paginated_absensi,
        karyawan=karyawan,
        page=page,
        total_pages=total_pages,
        start_index=start,
        bulan_filter=bulan_filter
    )


@app.route('/gaji_saya')
def gaji_saya():
    if 'karyawan' not in session:
        return redirect('/login_karyawan')

    karyawan = session['karyawan']
    id_karyawan = karyawan.get('id')

    # Ambil data karyawan
    karyawan_ref = db.reference(f'employees/{id_karyawan}')
    data_karyawan = karyawan_ref.get() or {}
    karyawan['nama'] = data_karyawan.get('nama', 'Tidak diketahui')
    karyawan['jabatan'] = data_karyawan.get('jabatan', 'Tidak diketahui')

    # Gaji default per hari
    gaji_per_hari = db.reference('default_gaji/default/gaji').get() or 0

    # Kasbon
    kasbon_ref = db.reference(f'kasbon/{id_karyawan}')
    kasbon_data = kasbon_ref.get() or {}
    kasbon_karyawan = 0
    total_kasbon = 0
    if isinstance(kasbon_data, dict):
        for val in kasbon_data.values():
            try:
                jumlah = int(val.get('jumlah', 0)) if isinstance(val, dict) else int(val)
                total_kasbon += jumlah
            except:
                continue
        kasbon_karyawan = total_kasbon

    # Data penggajian
    penggajian_ref = db.reference(f'penggajian/{id_karyawan}')
    data_penggajian = penggajian_ref.get() or {}
    tanggal_sudah_diambil = set()
    riwayat_gaji = []

    for key, item in data_penggajian.items():
        detail_list = item.get('detail', [])
        if isinstance(detail_list, dict):
            detail_list = [detail_list]
        elif not isinstance(detail_list, list):
            continue

        for detail in detail_list:
            status = detail.get('status', '-')
            try:
                total_gaji = int(item.get('total_gaji') or detail.get('gaji') or 0)
            except:
                total_gaji = 0

            try:
                total_kasbon_entry = int(item.get('total_kasbon', 0))
            except:
                total_kasbon_entry = 0

            # Tanggal gajian dalam format yang bisa dibandingkan ke attendance
            tanggal_str = detail.get('tanggal', '-')
            try:
                tanggal_gajian = datetime.strptime(tanggal_str, "%d %B %Y").strftime("%Y-%m-%d")
            except:
                tanggal_gajian = '-'

            if status == 'sudah diambil':
                tanggal_sudah_diambil.add(tanggal_gajian)

            try:
                sisa_gaji = int(detail.get('sisa_gaji', total_gaji - total_kasbon_entry))
            except:
                sisa_gaji = total_gaji - total_kasbon_entry

            try:
                tanggal_pengambilan = datetime.strptime(key.split('_')[0], "%Y%m%d").strftime("%d %B %Y")
            except:
                tanggal_pengambilan = '-'

            waktu = key.split('_')[1] if '_' in key else '-'

            riwayat_gaji.append({
                'tanggal_gajian': tanggal_gajian,
                'tanggal_pengambilan': tanggal_pengambilan,
                'waktu': waktu,
                'total_gaji': total_gaji,
                'kasbon': total_kasbon_entry,
                'sisa_gaji': sisa_gaji,
                'status': status
            })


    # Attendance Harian
    attendance_ref = db.reference('attendance_karyawan')
    attendance_data = attendance_ref.get() or {}
    presensi_harian = defaultdict(list)

    for jadwal_id, karyawan_data in attendance_data.items():
        if id_karyawan in karyawan_data:
            for record_id, detail in karyawan_data[id_karyawan].items():
                timestamp_str = detail.get('timestamp')
                try:
                    tgl_obj = datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S")
                    tanggal_str = tgl_obj.strftime("%Y-%m-%d")
                    presensi_harian[tanggal_str].append(tgl_obj)
                except:
                    continue

    # Tambahkan gaji belum diambil dari presensi (jika tanggal belum pernah digaji)
    gaji_belum_diambil = []
    for tanggal, entry_list in presensi_harian.items():
        if len(entry_list) >= 2 and tanggal not in tanggal_sudah_diambil:
            gaji_belum_diambil.append({
                'tanggal_gajian': tanggal,
                'tanggal_pengambilan': '-',
                'waktu': '-',
                'total_gaji': gaji_per_hari,
                'kasbon': 0,  # kasbon hanya dikurangkan sekali di akhir
                'sisa_gaji': gaji_per_hari,
                'status': 'belum diambil'
            })

    # Akumulasi total gaji dari yang belum diambil
    total_belum_diambil = sum(g['total_gaji'] for g in gaji_belum_diambil)
    if total_belum_diambil:
        sisa_belum_diambil = max(0, total_belum_diambil - kasbon_karyawan)
        riwayat_gaji.append({
            'tanggal_gajian': '-',
            'tanggal_pengambilan': '-',
            'waktu': '-',
            'total_gaji': total_belum_diambil,
            'kasbon': kasbon_karyawan,
            'sisa_gaji': sisa_belum_diambil,
            'status': 'belum diambil'
        })

    riwayat_gaji = sorted(riwayat_gaji, key=lambda x: x['status'].lower() != 'belum diambil')

    return render_template(
        'gaji_saya.html',
        karyawan=karyawan,
        gaji_list=riwayat_gaji,
        total_kasbon=total_kasbon
    )


# Fungsi untuk menyimpan progres
def save_progress(progress):
    with open('progress.json', 'w') as f:
        json.dump({'progress': progress}, f)

# Halaman Training Model
@app.route('/train', methods=['GET', 'POST'])
def train():
    if 'user' not in session:
        return redirect('/')

    if request.method == 'POST':
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
            import matplotlib.pyplot as plt
            from sklearn.metrics import confusion_matrix, classification_report
            import numpy as np
            import seaborn as sns
            import os
            import json

            # Konfigurasi dataset
            dataset_path = "D:/cobaf/AttendEaseMahasiswa/DataSet"
            test_dataset_path = "D:/cobaf/AttendEaseMahasiswa/DataTest"
            img_size = (224, 224)
            batch_size = 32

            # Augmentasi Data
            train_gen = ImageDataGenerator(
                rescale=1.0 / 255,
                validation_split=0.2,
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                brightness_range=[0.8, 1.2],
                horizontal_flip=True,
                fill_mode="nearest"
            )

            # Data training dan validasi
            train_data = train_gen.flow_from_directory(
                dataset_path,
                target_size=img_size,
                color_mode='rgb',
                batch_size=batch_size,
                class_mode='categorical',
                subset='training'
            )

            valid_data = train_gen.flow_from_directory(
                dataset_path,
                target_size=img_size,
                color_mode='rgb',
                batch_size=batch_size,
                class_mode='categorical',
                subset='validation'
            )

            # Jumlah kelas
            num_classes = train_data.num_classes

            # Model MobileNetV2
            mobilenet = tf.keras.applications.MobileNetV2(
                include_top=False,
                weights='imagenet',
                input_shape=(224, 224, 3)
            )

            for layer in mobilenet.layers[:-10]:
                layer.trainable = False

            # Tambahkan lapisan klasifikasi
            model = models.Sequential([
                mobilenet,
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation='softmax')
            ])

            # Kompilasi model
            model.compile(
                optimizer=Adam(learning_rate=0.00001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            # Callback untuk menyimpan progres
            class ProgressCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = logs.get('accuracy') * 100  # Simpan akurasi sebagai progres
                    save_progress(progress)  # Memanggil fungsi save_progress
                    print(f"Epoch {epoch + 1}: Progres = {progress}%")  # Log untuk debugging

            # Callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            checkpoint = ModelCheckpoint('models/best_finetuned_model_mobilenet.keras', monitor='val_accuracy', save_best_only=True)

            # Pelatihan model
            history = model.fit(
                train_data,
                validation_data=valid_data,
                epochs=30,
                callbacks=[early_stopping, checkpoint, ProgressCallback()]
            )

            # Simpan model akhir
            final_model_path = "models/finetuned_face_recognition_model_mobilenet.keras"
            model.save(final_model_path)

            # Simpan label map
            label_map = train_data.class_indices
            label_map = {str(int(v) + 1): k for k, v in label_map.items()}
            with open('label_map.json', 'w') as f:
                json.dump(label_map, f)

                      # ----- Evaluasi Model pada Data Testing -----
            print("----- Evaluasi Model pada Data Testing -----")

            # ImageDataGenerator untuk data testing tanpa augmentasi
            test_gen = ImageDataGenerator(rescale=1.0 / 255)

            # Data testing
            test_data = test_gen.flow_from_directory(
                test_dataset_path,
                target_size=img_size,
                color_mode='rgb',
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False
            )

            # Evaluasi model pada data testing
            test_loss, test_accuracy = model.evaluate(test_data)
            print("Loss pada Data Testing:", test_loss)
            print("Akurasi pada Data Testing:", test_accuracy)

            # Confusion Matrix untuk Data Testing
            y_pred_test = model.predict(test_data)
            y_pred_classes_test = np.argmax(y_pred_test, axis=1)
            y_true_test = test_data.classes

            cm_test = confusion_matrix(y_true_test, y_pred_classes_test)
            cm_labels_test = list(test_data.class_indices.keys())

            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels_test, yticklabels=cm_labels_test)
            plt.title('Confusion Matrix (Data Testing)')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.savefig('static/confusion_matrix_test.png')

            # ----- Confusion Matrix untuk Dataset Keseluruhan -----
            print("----- Confusion Matrix untuk Dataset Keseluruhan -----")
            full_data_gen = ImageDataGenerator(rescale=1.0 / 255)
            full_data = full_data_gen.flow_from_directory(
                dataset_path,
                target_size=img_size,
                color_mode='rgb',
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False
            )

            y_pred_full = model.predict(full_data)
            y_pred_classes_full = np.argmax(y_pred_full, axis=1)
            y_true_full = full_data.classes

            cm_full = confusion_matrix(y_true_full, y_pred_classes_full)
            cm_labels_full = list(full_data.class_indices.keys())

            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_full, annot=True, fmt='d', cmap='Oranges', xticklabels=cm_labels_full, yticklabels=cm_labels_full)
            plt.title('Confusion Matrix (Dataset Keseluruhan)')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.savefig('static/confusion_matrix_full.png')

            # Grafik Akurasi dan Loss
            plt.figure(figsize=(8, 6))
            plt.plot(history.history['accuracy'], label='Akurasi Training')
            plt.plot(history.history['val_accuracy'], label='Akurasi Validasi')
            plt.title('Akurasi Training dan Validasi')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid()
            plt.savefig('static/training_validation_accuracy.png')

            plt.figure(figsize=(8, 6))
            plt.plot(history.history['loss'], label='Loss Training')
            plt.plot(history.history['val_loss'], label='Loss Validasi')
            plt.title('Loss Training dan Validasi')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid()
            plt.savefig('static/training_validation_loss.png')

            # Classification Report
            report = classification_report(y_true_test, y_pred_classes_test, target_names=cm_labels_test, output_dict=True)
            report_path = 'static/classification_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f)

            return render_template(
                'train.html',
                message="Model berhasil dilatih dan dievaluasi!",
                test_accuracy=test_accuracy,
                test_loss=test_loss,
                confusion_matrix_path='static/confusion_matrix_test.png',
                full_confusion_matrix_path='static/confusion_matrix_full.png',
                accuracy_path='static/training_validation_accuracy.png',
                loss_path='static/training_validation_loss.png',
                report_path=report_path
            )

        except Exception as e:
            return render_template('train.html', message=f"Terjadi kesalahan saat pelatihan: {str(e)}")

    return render_template('train.html', message=None)

# Endpoint untuk polling progres
@app.route('/get_progress', methods=['GET'])
def get_progress():
    try:
        with open('progress.json', 'r') as f:
            progress_data = json.load(f)
        return jsonify(progress_data)
    except FileNotFoundError:
        return jsonify({'progress': 0})




@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('mahasiswa', None)
    return redirect('/')

# Jalankan aplikasi Flask
if __name__ == "__main__":
    app.run(debug=True, port=5050)

