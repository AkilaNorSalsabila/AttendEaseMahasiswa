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
import re  # Tambahkan baris ini
from flask import Flask, render_template, request, redirect, session, jsonify
from tensorflow.keras.models import load_model



app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Ganti dengan secret key yang aman
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # Maksimum 64 MB

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
dataset_path = 'DataSet'



# Inisialisasi Firebase
# Inisialisasi Firebase

# Load model hasil fine-tuning
model = tf.keras.models.load_model('models/best_finetuned_model_mobilenet.keras')

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

        # Validasi input
        if not username or not password:
            return render_template('register.html', message="Username dan Password harus diisi!")

        # Hash password sebelum menyimpannya
        hashed_password = generate_password_hash(password)

        try:
            # Simpan ke Firebase
            ref = db.reference('akun')
            ref.push({
                'username': username,
                'password': hashed_password
            })
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


# Memanmbahkan Dataset
# Memanmbahkan Dataset 
# Tambah route dataset
@app.route('/dataset', methods=['GET', 'POST'])
def dataset():
    if request.method == 'GET':
        return render_template('dataset.html')

    elif request.method == 'POST':
        try:
            # Ambil data dari form
            name = request.form.get('name')
            jabatan = request.form.get('jabatan')

            if not all([name, jabatan]):
                return jsonify({'status': 'error', 'message': 'Nama dan Jabatan harus diisi!'}), 400

            # Generate ID baru
            employee_id = generate_new_employee_id()

            # Siapkan folder lokal untuk menyimpan dataset
            folder_name = f"{employee_id}-{name.replace(' ', '_')}"
            folder_path = os.path.join(dataset_path, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            # Ambil file gambar dari request
            images = [request.form.get(key) for key in request.form if key.startswith('image_')]
            if not images:
                return jsonify({'status': 'error', 'message': 'Tidak ada gambar yang diterima!'}), 400
            def process_and_crop_faces(image, file_name_prefix, save_folder, user_id, user_name, start_count=0, padding=0.2):
                """
                Proses wajah: crop, resize, dan simpan lokal + Firebase Storage.
                """
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
            employee_ref = db.reference(f'employees/{employee_id}')
            employee_ref.set({
                'id': employee_id,
                'name': name,
                'id_karyawan': employee_id.split('-')[-1],  # hanya nomor belakang
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


# @app.route('/attendance', methods=['GET'])
# def attendance_report():
#     try:
#         employees_ref = db.collection('employees')
#         employees_docs = employees_ref.stream()

#         jadwal_ref = db.collection('jadwal_kerja')
#         jadwal_docs = jadwal_ref.stream()

#         attendance_ref = db.collection('attendance_karyawan')
#         attendance_docs = attendance_ref.stream()

#         # Ambil data ke dalam dictionary
#         employees = {doc.id: doc.to_dict() for doc in employees_docs}
#         jadwal_list = [doc.to_dict() for doc in jadwal_docs]
#         attendance_list = [doc.to_dict() for doc in attendance_docs]

#         # Buat laporan
#         report = []
#         for schedule in jadwal_list:
#             tanggal = schedule.get('tanggal')
#             for emp_id, emp_data in employees.items():
#                 attendance_record = next(
#                     (att for att in attendance_list if att.get('employee_id') == emp_id and att.get('tanggal') == tanggal),
#                     None
#                 )

#                 if attendance_record and attendance_record.get('jam_masuk') and attendance_record.get('jam_pulang'):
#                     status = 'Hadir'
#                     jam_masuk = attendance_record.get('jam_masuk')
#                     jam_pulang = attendance_record.get('jam_pulang')
#                     bukti = attendance_record.get('bukti')
#                 else:
#                     status = 'Tidak Hadir'
#                     jam_masuk = None
#                     jam_pulang = None
#                     bukti = None

#                 report.append({
#                     'id_karyawan': emp_id,
#                     'nama': emp_data.get('nama'),
#                     'tanggal': tanggal,
#                     'jam_masuk': jam_masuk,
#                     'jam_pulang': jam_pulang,
#                     'status': status,
#                     'bukti': bukti
#                 })

#         return jsonify(report), 200

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


@app.route("/attendance", methods=["GET", "POST"])
def attendance():
    ref = db.reference("attendance_karyawan")
    snapshot = ref.get()

    attendance_list = []

    if snapshot:
        for id_jadwal, karyawan_data in snapshot.items():
            for id_karyawan, records in karyawan_data.items():
                for record_id, details in records.items():
                    # parsing tanggal dari timestamp
                    try:
                        dt_object = datetime.strptime(details.get("timestamp", ""), "%Y-%m-%dT%H-%M-%S")
                        tanggal = dt_object.strftime("%Y-%m-%d")
                    except:
                        tanggal = "Invalid Date"

                    attendance_list.append({
                        "id_karyawan": id_karyawan,
                        "nama_karyawan":details.get("name", "-"),  # Kalau mau nama asli, update firebase
                        "tanggal": tanggal,
                        "jam_masuk": details.get("jam_masuk", "-"),
                        "jam_pulang": details.get("jam_pulang", "-"),
                        "status": ("Hadir" if details.get("jam_masuk") != "Tidak Diketahui" and details.get("jam_pulang") != "Tidak Diketahui"
                                   else "Tidak Hadir"),
                        "image_url": details.get("image_url", ""),
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

if __name__ == "__main__":
    app.run(debug=True)
@app.route('/admin/penggajian', methods=['GET', 'POST'])
def admin_penggajian():
    if 'user' not in session:
        return redirect('/login_admin')

    # Inisialisasi variabel
    nama_karyawan = request.form.get('nama_karyawan', '')
    tanggal_mulai = request.form.get('tanggal_mulai', '')
    tanggal_selesai = request.form.get('tanggal_selesai', '')
    attendance_list = []

    # Ambil data dari Firebase
    attendance_ref = db.reference('attendance')
    attendance_data = attendance_ref.get() or {}

    employees_ref = db.reference('employees')
    employees_data = employees_ref.get() or {}

    # 🔵 Buat daftar nama karyawan dari employees dan attendance (gabungan)
    nama_karyawan_set = set()

    # Dari employees
    for emp_id, emp_info in employees_data.items():
        nama = emp_info.get('nama', '-')
        if nama != "-":
            nama_karyawan_set.add(nama)

    # Dari attendance
    for jadwal_db, minggu_data in attendance_data.items():
        for minggu_ke, employee_data in minggu_data.items():
            for emp_id, records in employee_data.items():
                for record_id, detail in records.items():
                    nama_raw = detail.get("name", "-")
                    if nama_raw and nama_raw != "-":
                        nama = nama_raw.split('-')[-1].strip()
                        nama_karyawan_set.add(nama)

    nama_karyawan_list = sorted(list(nama_karyawan_set))  # Urutkan biar rapi

    # 🔵 Fungsi untuk mengubah string tanggal menjadi objek datetime
    def str_to_date(date_str):
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return None

    # 🔵 Konversi tanggal mulai dan selesai ke datetime
    tanggal_mulai_obj = str_to_date(tanggal_mulai) if tanggal_mulai else None
    tanggal_selesai_obj = str_to_date(tanggal_selesai) if tanggal_selesai else None

    # 🔵 Proses data absensi
    for jadwal_db, minggu_data in attendance_data.items():
        for minggu_ke, employee_data in minggu_data.items():
            for emp_id, records in employee_data.items():
                for record_id, detail in records.items():
                    nama_raw = detail.get("name", "-")
                    nama = nama_raw.split('-')[-1].strip() if nama_raw and nama_raw != "-" else "-"

                    timestamp_str = detail.get("timestamp", "-")
                    
                    # Convert timestamp to datetime object
                    try:
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S') if timestamp_str != '-' else None
                    except ValueError:
                        timestamp = None

                    # Filter berdasarkan nama karyawan (kalau ada)
                    if nama_karyawan and nama != nama_karyawan:
                        continue
                    
                    # Filter berdasarkan rentang tanggal (kalau ada)
                    if tanggal_mulai_obj and timestamp and timestamp < tanggal_mulai_obj:
                        continue
                    if tanggal_selesai_obj and timestamp and timestamp > tanggal_selesai_obj:
                        continue

                    status = detail.get("status", "Hadir")
                    gaji = 100000 if status == "Hadir" else 0
                    minggu_number = re.sub(r'\D', '', minggu_ke)

                    attendance_list.append({
                        "id_karyawan": emp_id,
                        "nama": nama,
                        "status": status,
                        "timestamp": timestamp_str,
                        "minggu_ke": minggu_number,
                        "gaji": gaji
                    })

    # 🔵 Urutkan berdasarkan minggu dan nama
    attendance_list = sorted(attendance_list, key=lambda x: (x['minggu_ke'], x['nama']))

    return render_template('penggajian.html',
                           attendance_list=attendance_list,
                           nama_karyawan_list=nama_karyawan_list,
                           nama_karyawan=nama_karyawan)

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

# @app.route('/students/delete/<student_id>', methods=['DELETE'])
# def delete_student(student_id):
#     """
#     Endpoint untuk menghapus data mahasiswa berdasarkan student_id.
#     """
#     try:
#         student_ref = db.reference(f'students/{student_id}')
        
#         # Hapus data mahasiswa dari Firebase
#         student_ref.delete()

#         return jsonify({'status': 'success', 'message': f'Data mahasiswa dengan ID {student_id} berhasil dihapus.'})
#     except Exception as e:
#         print(f"Error saat menghapus data mahasiswa: {str(e)}")
#         return jsonify({'status': 'error', 'message': str(e)}), 500

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


# # Route untuk mengambil data mahasiswa
# @app.route('/students', methods=['GET'])
# def get_students():
#     try:
#         combined_data = []

#         # Ambil data mahasiswa
#         students_ref = db.reference('students')
#         students_data = students_ref.get()
#         if students_data:
#             for student_id, student_info in students_data.items():
#                 folder_path = os.path.join(dataset_path, student_id)
#                 if os.path.exists(folder_path):
#                     images_count = len([
#                         f for f in os.listdir(folder_path)
#                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))
#                     ])
#                     if 'images_count' not in student_info or student_info['images_count'] == 0:
#                         student_ref = db.reference(f'students/{student_id}')
#                         student_ref.update({'images_count': images_count})

#                 combined_data.append({
#                     'id': student_id,
#                     'name': student_info.get('name', 'Unknown'),
#                     'golongan': student_info.get('golongan', 'Unknown'),
#                     'semester': student_info.get('semester', ''),
#                     'jabatan': '-',  # kosongkan karena mahasiswa
#                     'images_count': student_info.get('images_count', 0),
#                     'edit_url': f'/students/edit/{student_id}',
#                     'delete_url': f'/students/delete/{student_id}'
#                 })

#         # Ambil data karyawan
#         employees_ref = db.reference('employees')
#         employees_data = employees_ref.get()
#         if employees_data:
#             for emp_id, emp_info in employees_data.items():
#                 folder_path = os.path.join(dataset_path, emp_id)
#                 if os.path.exists(folder_path):
#                     images_count = len([
#                         f for f in os.listdir(folder_path)
#                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))
#                     ])
#                     if 'images_count' not in emp_info or emp_info['images_count'] == 0:
#                         emp_ref = db.reference(f'employees/{emp_id}')
#                         emp_ref.update({'images_count': images_count})

#                 combined_data.append({
#                     'id': emp_id,
#                     'name': emp_info.get('name', 'Unknown'),
#                     'golongan': '-',  # kosongkan karena karyawan
#                     'semester': '-',  # kosongkan karena karyawan
#                     'jabatan': emp_info.get('jabatan', 'Unknown'),
#                     'images_count': emp_info.get('images_count', 0),
#                     'edit_url': f'/students/edit/{emp_id}',  # kamu bisa sesuaikan endpoint edit-nya
#                     'delete_url': f'/students/delete/{emp_id}'  # kamu bisa sesuaikan juga
#                 })

#         return jsonify({'status': 'success', 'data': combined_data})

#     except Exception as e:
#         print(f"Error saat mengambil data: {str(e)}")
#         return jsonify({'status': 'error', 'message': str(e)}), 500

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
    
# @app.route('/absen', methods=['GET'])
# def absen():
#     if 'karyawan' not in session:
#         flash('Silakan login terlebih dahulu', 'warning')
#         return redirect('/login_karyawan')

#     karyawan = session['karyawan']
    
#     return render_template('absen.html', karyawan=karyawan)

# def run_face_recognition(user_id):
#     """
#     Fungsi untuk deteksi wajah dan menyimpan data absensi.
#     """
#     username_full = labels.get(user_id, "Unknown")
#     username = username_full.split("", 1)[1] if "" in username_full else username_full

#     video = cv2.VideoCapture(0)
#     if not video.isOpened():
#         print("[ERROR] Kamera tidak dapat dibuka. Pastikan kamera tersedia.")
#         return

#     face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#     locked_label = "Unknown"
#     lock_frames = 0
#     threshold_confidence = 0.75
#     min_consecutive_frames = 5
#     attendance_logged = False

#     print("[INFO] Mulai deteksi wajah...")
#     while True:
#         ret, frame = video.read()
#         if not ret:
#             print("[ERROR] Tidak dapat membaca frame dari kamera.")
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#         if len(faces) == 0:
#             print("[INFO] Tidak ada wajah yang terdeteksi.")
#         else:
#             print(f"[INFO] {len(faces)} wajah terdeteksi.")

#         for (x, y, w, h) in faces:
#             face_img = frame[y:y + h, x:x + w]
#             face_img = cv2.resize(face_img, (224, 224))
#             face_img = np.expand_dims(face_img, axis=0) / 255.0

#             prediction = model.predict(face_img)
#             confidence = float(np.max(prediction[0]))
#             id_detected = str(np.argmax(prediction[0]) + 1)

#             print(f"[DEBUG] Deteksi: ID={id_detected}, Confidence={confidence:.2f}, Lock frames={lock_frames}")

#             if confidence >= threshold_confidence and id_detected == user_id:
#                 lock_frames += 1
#                 if lock_frames >= min_consecutive_frames and not attendance_logged:
#                     try:
#                         # Simpan frame sebagai gambar
#                         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                         image_path = f"attendance_{user_id}_{timestamp}.jpg"
#                         cv2.imwrite(image_path, frame)
#                         print("[INFO] Gambar berhasil disimpan:", image_path)

#                         # Unggah gambar ke Firebase Storage
#                         try:
#                             blob = bucket.blob(f'attendance_images/{image_path}')
#                             blob.upload_from_filename(image_path)
#                             blob.make_public()
#                             image_url = blob.public_url
#                             print("[SUCCESS] Gambar berhasil diunggah ke Firebase Storage. URL:", image_url)
#                         except Exception as e:
#                             print("[ERROR] Gagal mengunggah gambar ke Firebase Storage:", str(e))
#                             continue

#                         # Simpan data ke Firebase Realtime Database
#                         try:
#                             attendance_data = {
#                                 'user_id': user_id,
#                                 'username': username,
#                                 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                                 'confidence': confidence,
#                                 'image_url': image_url
#                             }
#                             db.reference(f'attendance/{user_id}').push(attendance_data)
#                             print("[SUCCESS] Data presensi berhasil disimpan:", attendance_data)
#                         except Exception as e:
#                             print("[ERROR] Gagal menyimpan data ke Firebase Realtime Database:", str(e))
#                             continue

#                         # Hapus gambar lokal
#                         os.remove(image_path)
#                         print("[INFO] Gambar lokal dihapus:", image_path)

#                         attendance_logged = True  # Tandai presensi sudah tercatat
#                         break  # Keluar dari loop setelah presensi berhasil
#                     except Exception as e:
#                         print("[ERROR] Terjadi kesalahan saat mencatat presensi:", str(e))
#             else:
#                 lock_frames = 0

#         if attendance_logged:
#                  print("[INFO] Presensi selesai, keluar dari loop.")
#                  break

#     video.release()
#     cv2.destroyAllWindows()
#     if not attendance_logged:
#         print("[ERROR] Presensi tidak tercatat. Pastikan wajah terlihat jelas.")

@app.route('/rekap_absensi', methods=['GET'])
def rekap_absensi():
    if 'mahasiswa' not in session:
        return redirect('/login_mahasiswa')

    mahasiswa = session['mahasiswa']
    attendance_ref = db.reference("attendance")
    attendance_data = attendance_ref.get() or {}

    mahasiswa_attendance = []

    # Iterasi data absensi
    for kode_mata_kuliah, minggu_data in attendance_data.items():
        # Pastikan minggu_data adalah dictionary atau list
        if isinstance(minggu_data, dict):
            for minggu, student_data in minggu_data.items():
                # Cek apakah student_data bukan None
                if student_data and mahasiswa['id'] in student_data:
                    # Masuk ke key acak
                    for random_key, detail in student_data[mahasiswa['id']].items():
                        # Cek apakah detail adalah dictionary sebelum memanggil .get()
                        if isinstance(detail, dict):
                            mahasiswa_attendance.append({
                                "kode_mata_kuliah": detail.get("kode_mata_kuliah", "Unknown"),
                                "nama_mata_kuliah": detail.get("nama_mata_kuliah", "Unknown"),
                                "minggu": minggu,
                                "timestamp": detail.get("timestamp", "Tidak Ada Data"),
                                "status": detail.get("status", "Tidak Hadir"),
                                "image_url": detail.get("image_url", None)
                            })
                        else:
                            # Jika detail bukan dictionary, tampilkan pesan debug
                            print(f"[DEBUG] Detail for {random_key} is not a dictionary: {detail}")

        elif isinstance(minggu_data, list):
            # Handle jika minggu_data adalah list
            for index, student_data in enumerate(minggu_data):
                # Cek apakah student_data bukan None
                if student_data and mahasiswa['id'] in student_data:
                    for random_key, detail in student_data[mahasiswa['id']].items():
                        if isinstance(detail, dict):
                            mahasiswa_attendance.append({
                                "kode_mata_kuliah": detail.get("kode_mata_kuliah", "Unknown"),
                                "nama_mata_kuliah": detail.get("nama_mata_kuliah", "Unknown"),
                                "minggu": index,  # Menggunakan index minggu dari list
                                "timestamp": detail.get("timestamp", "Tidak Ada Data"),
                                "status": detail.get("status", "Tidak Hadir"),
                                "image_url": detail.get("image_url", None)
                            })
                        else:
                            # Jika detail bukan dictionary, tampilkan pesan debug
                            print(f"[DEBUG] Detail for {random_key} is not a dictionary: {detail}")

    # Debug data absensi mahasiswa
    print(f"[DEBUG] Mahasiswa Attendance: {mahasiswa_attendance}")

    return render_template(
        'rekap_absensi.html',
        mahasiswa=mahasiswa,
        attendance_list=mahasiswa_attendance
    )


# @app.route('/check_absen_status', methods=['GET'])
# def check_absen_status():
#     # Mendapatkan parameter dari URL
#     user_id = request.args.get('user_id')
#     jadwal_kerja = request.args.get('jadwal_kerja')  # Harus berupa kode asli
#     minggu_ke = request.args.get('minggu_ke')

#     # Validasi parameter
#     if not user_id or not jadwal_kerja or not minggu_ke:
#         print("[ERROR] Parameter tidak lengkap. Pastikan 'user_id', 'jadwal_kerja', dan 'minggu_ke' disertakan.")
#         return jsonify({"status": "error", "message": "Parameter tidak lengkap"}), 400

#     print(f"[DEBUG] Checking attendance for user_id={user_id}, mata_kuliah={jadwal_kerja}, minggu_ke={minggu_ke}")

#     try:
#         # Referensi ke lokasi data absensi di Firebase Realtime Database
#         attendance_ref = db.reference(f"attendance/{jadwal_kerja}/{minggu_ke}/{user_id}")
#         data = attendance_ref.get()

#         if data:
#             print("[DEBUG] Attendance Data Found:", data)
#             return jsonify({"status": "success", "data": data})
#         else:
#             print("[DEBUG] Attendance Data Not Found")
#             return jsonify({"status": "pending", "message": "Data absensi tidak ditemukan"}), 404
#     except Exception as e:
#         # Menangkap kesalahan selama proses membaca data dari Firebase
#         print(f"[ERROR] Terjadi kesalahan saat memeriksa status absensi: {e}")
#         return jsonify({"status": "error", "message": "Terjadi kesalahan server"}), 500

# @app.route('/check_absen_status_karyawan', methods=['GET'])
# def check_absen_status_karyawan():
#     user_id = request.args.get('user_id')
#     jadwal_id = request.args.get('jadwal_id')

#     if not user_id or not jadwal_id:
#         print("[ERROR] Parameter tidak lengkap. Pastikan 'user_id' dan 'jadwal_id' disertakan.")
#         return jsonify({"status": "error", "message": "Parameter tidak lengkap"}), 400

#     print(f"[DEBUG] Checking attendance for user_id={user_id}, jadwal_id={jadwal_id}")

#     try:
#         # Referensi ke database karyawan
#         attendance_ref = db.reference(f"attendance_karyawan/{jadwal_id}/{user_id}")
#         data = attendance_ref.get()

#         if data:
#             print("[DEBUG] Attendance Data Found:", data)
#             return jsonify({"status": "success", "data": data})
#         else:
#             print("[DEBUG] Attendance Data Not Found")
#             return jsonify({"status": "pending", "message": "Data absensi tidak ditemukan"}), 404
#     except Exception as e:
#         print(f"[ERROR] Terjadi kesalahan saat memeriksa status absensi: {e}")
#         return jsonify({"status": "error", "message": "Terjadi kesalahan server"}), 500

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
