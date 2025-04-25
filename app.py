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
# ... import lainnya yang sudah ada



app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Ganti dengan secret key yang aman
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # Maksimum 64 MB

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
dataset_path = 'DataSet'



# Inisialisasi Firebase
# Inisialisasi Firebase
cred = credentials.Certificate("D:/coba/facerecognition-c8264-firebase-adminsdk-nodyk-90850d2e73.json")
initialize_app(cred, {
    'databaseURL': 'https://facerecognition-c8264-default-rtdb.firebaseio.com/',
})
bucket = storage.bucket('facerecognition-c8264.appspot.com')

# Load model hasil fine-tuning
model = tf.keras.models.load_model('models/best_finetuned_model_mobilenet.keras')

# Load label dari file JSON
with open('label_map.json', 'r') as f:
    labels = json.load(f)

# Path dataset
dataset_path = 'DataSet'

test_dataset_path = "DataTest"

faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
# Fungsi untuk mengunggah dataset manual ke Firebase
# Fungsi untuk mencatat metadata ke Firebase
# Fungsi untuk mencatat metadata ke Firebase, termasuk jumlah gambar


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

@app.route('/karyawan_dashboard')
def karyawan_dashboard():
    if 'karyawan' not in session:
        flash('Silakan login terlebih dahulu', 'warning')
        return redirect('/login_karyawan')

    karyawan = session['karyawan']  # Data karyawan
    jabatan = karyawan.get('jabatan', '')  # Mengambil jabatan dari session

    # Ambil daftar jadwal kerja berdasarkan jabatan
    jadwal_kerja_data = []
    jadwal_ref = db.reference(f'jadwal_kerja/{jabatan}')
    jadwal_data = jadwal_ref.get()

    # Menambahkan log untuk debugging
    print(f"[DEBUG] Data jadwal kerja yang diterima: {jadwal_data}")

    # Pastikan jadwal_data memiliki data dan proses dengan aman
    if isinstance(jadwal_data, dict):  # Memastikan bahwa jadwal_data adalah dictionary
        for k, v in jadwal_data.items():
            if isinstance(v, dict):  # Memastikan v adalah dictionary
                jadwal_kerja_data.append({
                    'id': k,
                    'name': v.get('name', 'Tidak Ada Nama'),
                    'jam_masuk': v.get('jam_masuk', 'Tidak Diketahui'),
                    'jam_pulang': v.get('jam_pulang', 'Tidak Diketahui'),
                    'status': v.get('status', 'Tidak Hadir')  # Menambahkan status jika ada
                })
            else:
                print(f"[WARNING] Data jadwal kerja tidak dalam format yang diharapkan: {v}")
    else:
        print("[ERROR] Data jadwal kerja kosong atau tidak dalam format yang benar.")

    # Kirim data karyawan dan jadwal_kerja_data ke template
    return render_template('karyawan_dashboard.html', karyawan=karyawan, jadwal_kerja_data=jadwal_kerja_data)

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

import time


def gen(user_id, mata_kuliah, minggu_ke, nim):
    golongan = "A"  # Golongan bisa disesuaikan sesuai kebutuhan
    jadwal_ref = db.reference(f'jadwal_mata_kuliah/{golongan}/{mata_kuliah}')
    jadwal_data = jadwal_ref.get()
    nama_mata_kuliah = jadwal_data.get('name', 'Unknown')
    kode_mata_kuliah_asli = jadwal_data.get('kode_mk', mata_kuliah)

    # Inisialisasi kamera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("[ERROR] Kamera tidak dapat dibuka. Pastikan kamera tersedia.")
        return

    attendance_logged = False
    detected_name = "Unknown"
    capture_dir = "captures"
    os.makedirs(capture_dir, exist_ok=True)

    try:
        while True:
            # Jika absensi sudah dicatat, hentikan generator
            if attendance_logged:
                print("[DEBUG] Absensi sudah dicatat. Menghentikan generator.")
                break

            success, frame = camera.read()
            if not success:
                print("[ERROR] Tidak dapat membaca frame dari kamera.")
                break

            # Konversi ke grayscale untuk deteksi wajah
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                # Preprocess wajah untuk input model
                face_img = frame[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, (224, 224))
                face_img = np.expand_dims(face_img, axis=0) / 255.0

                # Prediksi menggunakan model CNN
                prediction = model.predict(face_img)
                confidence = float(np.max(prediction[0]))
                detected_id = str(np.argmax(prediction[0]) + 1)
                detected_name = labels.get(detected_id, "Unknown")

                print(f"[DEBUG] Deteksi wajah: ID={detected_id}, Confidence={confidence:.2f}, Nama={detected_name}")

                # Validasi NIM
                detected_nim = detected_name.split("-")[1] if "-" in detected_name else None
                expected_nim = user_id.split("-")[1] if "-" in user_id else None

                print(f"[DEBUG] Detected NIM: {detected_nim}, Expected NIM: {expected_nim}")

                # Tambahkan kotak di sekitar wajah dan teks
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{detected_name} ({confidence * 100:.2f}%)",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # Cek apakah absensi dapat dicatat
                if confidence >= 0.75 and detected_nim == expected_nim and not attendance_logged:
                    try:
                        # Simpan absensi ke Firebase
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        image_path = f"attendance_{user_id}_{timestamp.replace(':', '-')}.jpg"
                        local_filepath = os.path.join(capture_dir, image_path)
                        cv2.imwrite(local_filepath, frame)

                        # Unggah ke Firebase Storage
                        blob = bucket.blob(f'attendance_images/{image_path}')
                        blob.upload_from_filename(local_filepath)
                        blob.make_public()
                        image_url = blob.public_url

                        # Simpan data absensi ke Realtime Database
                        attendance_ref = db.reference(f"attendance/{mata_kuliah}/{minggu_ke}/{user_id}")
                        attendance_data = {
                            "nim": nim,
                            "name": detected_name,
                            "kode_mata_kuliah": mata_kuliah,
                            "nama_mata_kuliah": nama_mata_kuliah,
                            "minggu_ke": minggu_ke,
                            "status": "Hadir",
                            "timestamp": timestamp,
                            "image_url": image_url,
                            "golongan": golongan
                        }
                        attendance_ref.push(attendance_data)
                        print(f"[SUCCESS] Data absensi berhasil disimpan: {attendance_data}")

                        attendance_logged = True
                        break
                    except Exception as e:
                        print(f"[ERROR] Terjadi kesalahan saat mencatat absensi: {e}")

            # Streaming frame ke frontend
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan pada fungsi gen: {e}")
    finally:
        # Pastikan kamera ditutup dan window dihentikan
        camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Kamera ditutup dan jendela video dihentikan.")


@app.route('/video_feed/<user_id>/<jadwal_kerja>/<minggu_ke>/<id>')
def video_feed(user_id, jadwal_kerja, minggu_ke,id):
    print(f"[DEBUG] Video feed accessed for user_id={user_id}, jadwal_kerja={jadwal_kerja}, minggu_ke={minggu_ke}, id={id}")
    return Response(
        gen(user_id, jadwal_kerja, minggu_ke, id),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


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



# @app.route('/attendance', methods=['GET', 'POST'])
# def admin_attendance():
#     """
#     Admin melihat laporan absensi berdasarkan golongan dan mata kuliah.
#     """
#     if 'user' not in session:
#         return redirect('/login_admin')

#     golongan = None
#     mata_kuliah = None
#     attendance_list = []

#     # Ambil data jadwal terlebih dahulu
#     jadwal_ref = db.reference('jadwal_mata_kuliah')
#     jadwal_data = jadwal_ref.get() or {}

#     # Pastikan jadwal_data terisi, jika kosong atau None, tangani dengan cara yang sesuai
#     if not jadwal_data:
#         jadwal_data = {}

#     # Jika form disubmit, ambil golongan dan mata kuliah dari form
#     if request.method == 'POST':
#         golongan = request.form.get('golongan')
#         mata_kuliah = request.form.get('mata_kuliah')

#         print(f"Golongan yang dipilih: {golongan}")  # Debugging output golongan
#         print(f"Mata Kuliah yang dipilih: {mata_kuliah}")  # Debugging output mata kuliah

#         # Ambil data dari Firebase berdasarkan golongan dan mata kuliah
#         attendance_ref = db.reference('attendance')
#         attendance_data = attendance_ref.get() or {}

#         # Ambil data jadwal untuk golongan dan mata kuliah yang dipilih
#         golongan_mahasiswa = jadwal_data.get(golongan, {}).get(mata_kuliah, [])

#         # Ambil data mahasiswa dari koleksi students
#         students_ref = db.reference('students')
#         students_data = students_ref.get() or {}

#         # Pastikan students_data adalah dictionary, bukan string
#         if isinstance(students_data, str):
#             students_data = {}

#         # Ambil daftar mahasiswa berdasarkan golongan yang dipilih
#         golongan_students = [student for student in students_data.values() if student['golongan'] == golongan]

#         # Proses data absensi sesuai golongan dan mata kuliah yang dipilih
#         for mata_kuliah_db, minggu_data in attendance_data.items():
#             if mata_kuliah_db != mata_kuliah:
#                 continue
#             for minggu_ke, student_data in minggu_data.items():
#                 for student_id, records in student_data.items():
#                     for record_id, detail in records.items():
#                         if isinstance(detail, dict) and detail.get("golongan") == golongan:
#                             full_name = detail.get("name", "Tidak Ada")
#                             name_only = full_name.split('-')[-1].strip() if full_name else "Tidak Ada"
                            
#                             # Gunakan regex untuk memastikan minggu_ke hanya berisi angka
#                             minggu_number = re.sub(r'\D', '', minggu_ke)  # Hapus semua karakter non-digit

#                             # Tambahkan data absensi mahasiswa yang hadir
#                             attendance_list.append({
#                                 "kode_mata_kuliah": detail.get("kode_mata_kuliah", "Tidak Ada"),
#                                 "nama_mata_kuliah": detail.get("nama_mata_kuliah", "Tidak Ada"),
#                                 "minggu_ke": int(minggu_number),  # Menggunakan minggu_ke sebagai integer
#                                 "nim": detail.get("nim", "Tidak Ada"),
#                                 "nama": name_only,
#                                 "status": detail.get("status", "Hadir"),
#                                 "timestamp": detail.get("timestamp", "Tidak Ada"),
#                                 "image_url": detail.get("image_url", None)
#                             })

#         # Menambahkan mahasiswa yang tidak hadir berdasarkan golongan
#         for mahasiswa in golongan_students:
#             found = False
#             for attendance in attendance_list:
#                 if attendance['nim'] == mahasiswa['nim']:  # Cek berdasarkan NIM
#                     found = True
#                     break

#             # Jika mahasiswa tidak ada dalam data absensi, tambahkan sebagai tidak hadir
#             if not found:
#                 # Tentukan minggu yang sesuai, gunakan minggu yang ada di data absensi
#                 for mata_kuliah_db, minggu_data in attendance_data.items():
#                     if mata_kuliah_db == mata_kuliah:
#                         for minggu_ke, student_data in minggu_data.items():
#                             # Gunakan minggu_ke dari data yang ada
#                             minggu_number = re.sub(r'\D', '', minggu_ke)  # Hapus semua karakter non-digit

#                             attendance_list.append({
#                                 "kode_mata_kuliah": mata_kuliah,
#                                 "nama_mata_kuliah": mata_kuliah,
#                                 "minggu_ke": int(minggu_number),  # Gunakan minggu_ke sebagai integer
#                                 "nim": mahasiswa['nim'],
#                                 "nama": mahasiswa['name'],
#                                 "status": "Tidak Hadir",
#                                 "timestamp": "Tidak Ada",
#                                 "image_url": None
#                             })

#         # Urutkan data berdasarkan minggu_ke dan nama mahasiswa
#         attendance_list = sorted(attendance_list, key=lambda x: (x['minggu_ke'], x['nama']))

#     # Ambil daftar golongan dari jadwal_data setelah memastikan data ada
#     golongan_list = list(jadwal_data.keys())  # Ambil daftar golongan (A, B, C)

#     # Ambil mata kuliah berdasarkan golongan yang dipilih
#     mata_kuliah_list = []
#     if golongan:
#         mata_kuliah_list = list(jadwal_data.get(golongan, {}).keys())

#     return render_template('attendance.html', 
#                            attendance_list=attendance_list,
#                            golongan_list=golongan_list,
#                            mata_kuliah_list=mata_kuliah_list,
#                            golongan=golongan, mata_kuliah=mata_kuliah)


# @app.route('/update_attendance', methods=['POST'])
# def update_attendance():
#     """
#     Admin memperbarui status absensi mahasiswa.
#     """
#     if 'user' not in session:
#         return redirect('/login_admin')

#     # Pastikan data yang diterima adalah JSON
#     data = request.get_json()
#     if not data:
#         return jsonify({'status': 'error', 'message': 'Invalid JSON data'}), 400
    
#     # Debugging data yang diterima
#     print(f"Data yang diterima: {data}")

#     nim = data.get('nim')
#     minggu_ke = data.get('minggu_ke')
#     status = data.get('status')

#     # Pastikan semua data yang diperlukan ada
#     if not nim or not minggu_ke or not status:
#         return jsonify({'status': 'error', 'message': 'NIM, Minggu Ke, dan Status harus ada.'}), 400

#     # Update status absensi di Firebase
#     attendance_ref = db.reference('attendance')
#     attendance_data = attendance_ref.get()

#     # Pastikan data absensi ada
#     if not attendance_data:
#         return jsonify({'status': 'error', 'message': 'Data absensi tidak ditemukan.'}), 404

#     # Cari entri yang sesuai dan update status
#     for mata_kuliah, minggu_data in attendance_data.items():
#         for minggu, student_data in minggu_data.items():
#             for student_id, records in student_data.items():
#                 if student_id == nim:  # Cari berdasarkan NIM
#                     for record_id, record_detail in records.items():
#                         if minggu == minggu_ke:  # Cari berdasarkan minggu
#                             record_detail['status'] = status  # Update status
#                             # Simpan perubahan ke Firebase
#                             attendance_ref.child(mata_kuliah).child(minggu).child(student_id).child(record_id).update(record_detail)
#                             return jsonify({'status': 'success', 'message': 'Absensi berhasil diperbarui.'})

#     return jsonify({'status': 'error', 'message': 'Data absensi tidak ditemukan untuk NIM dan minggu yang diberikan.'}), 404
#KARYAWAN ATTENDANCE
@app.route('/attendance', methods=['GET', 'POST'])
def admin_attendance():
    if 'user' not in session:
        return redirect('/login_admin')

    # Inisialisasi variabel
    jabatan = request.form.get('jabatan', '')
    jadwal_kerja = request.form.get('jadwal_kerja', '')
    attendance_list = []

    # Ambil semua data yang diperlukan
    jadwal_ref = db.reference('jadwal_kerja')
    jadwal_data = jadwal_ref.get() or {}
    jabatan_list = list(jadwal_data.keys())

    # Ambil data attendance dan employees sekaligus
    attendance_ref = db.reference('attendance')
    attendance_data = attendance_ref.get() or {}
    
    employees_ref = db.reference('employees')
    employees_data = employees_ref.get() or {}

    # Jika ada filter jabatan, ambil jadwal kerja yang sesuai
    jadwal_kerja_list = []
    if jabatan:
        jadwal_kerja_list = list(jadwal_data.get(jabatan, {}).keys())
    else:
        # Jika tidak ada filter, ambil semua jadwal kerja
        for jb in jabatan_list:
            jadwal_kerja_list.extend(list(jadwal_data.get(jb, {}).keys()))

    # Proses data absensi (tanpa filter awal)
    for jadwal_db, minggu_data in attendance_data.items():
        # Skip jika ada filter jadwal_kerja dan tidak match
        if jadwal_kerja and jadwal_db != jadwal_kerja:
            continue
            
        for minggu_ke, employee_data in minggu_data.items():
            for emp_id, records in employee_data.items():
                for record_id, detail in records.items():
                    # Skip jika ada filter jabatan dan tidak match
                    if jabatan and detail.get("jabatan") != jabatan:
                        continue
                        
                    minggu_number = re.sub(r'\D', '', minggu_ke)
                    attendance_list.append({
                        "kode_jadwal_kerja": detail.get("kode_jadwal_kerja", "-"),
                        "jadwal_kerja": jadwal_db,
                        "minggu_ke": int(minggu_number),
                        "id_karyawan": emp_id,
                        "nama": detail.get("name", "-").split('-')[-1].strip(),
                        "status": detail.get("status", "Hadir"),
                        "timestamp": detail.get("timestamp", "-"),
                        "image_url": detail.get("image_url")
                    })

    # Urutkan data
    attendance_list = sorted(attendance_list, key=lambda x: (x['minggu_ke'], x['nama']))

    return render_template('attendance.html',
                        attendance_list=attendance_list,
                        jabatan_list=jabatan_list,
                        jadwal_kerja_list=jadwal_kerja_list,
                        jabatan=jabatan,
                        jadwal_kerja=jadwal_kerja)

@app.route('/students/edit/<student_id>', methods=['POST'])
def edit_student(student_id):
    """
    Endpoint untuk mengedit data mahasiswa berdasarkan student_id.
    """
    try:
        data = request.json  # Data dikirim dalam format JSON dari frontend
        student_ref = db.reference(f'students/{student_id}')
        
        # Perbarui data mahasiswa di Firebase
        updated_data = {
            'semester': data.get('semester', ''),
            'golongan': data.get('golongan', '')
        }
        student_ref.update(updated_data)

        return jsonify({'status': 'success', 'message': f'Data mahasiswa dengan ID {student_id} berhasil diperbarui.'})
    except Exception as e:
        print(f"Error saat memperbarui data mahasiswa: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/students/delete/<student_id>', methods=['DELETE'])
def delete_student(student_id):
    """
    Endpoint untuk menghapus data mahasiswa berdasarkan student_id.
    """
    try:
        student_ref = db.reference(f'students/{student_id}')
        
        # Hapus data mahasiswa dari Firebase
        student_ref.delete()

        return jsonify({'status': 'success', 'message': f'Data mahasiswa dengan ID {student_id} berhasil dihapus.'})
    except Exception as e:
        print(f"Error saat menghapus data mahasiswa: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# Route untuk mengambil data mahasiswa
@app.route('/students', methods=['GET'])
def get_students():
    try:
        combined_data = []

        # Ambil data mahasiswa
        students_ref = db.reference('students')
        students_data = students_ref.get()
        if students_data:
            for student_id, student_info in students_data.items():
                folder_path = os.path.join(dataset_path, student_id)
                if os.path.exists(folder_path):
                    images_count = len([
                        f for f in os.listdir(folder_path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                    ])
                    if 'images_count' not in student_info or student_info['images_count'] == 0:
                        student_ref = db.reference(f'students/{student_id}')
                        student_ref.update({'images_count': images_count})

                combined_data.append({
                    'id': student_id,
                    'name': student_info.get('name', 'Unknown'),
                    'golongan': student_info.get('golongan', 'Unknown'),
                    'semester': student_info.get('semester', ''),
                    'jabatan': '-',  # kosongkan karena mahasiswa
                    'images_count': student_info.get('images_count', 0),
                    'edit_url': f'/students/edit/{student_id}',
                    'delete_url': f'/students/delete/{student_id}'
                })

        # Ambil data karyawan
        employees_ref = db.reference('employees')
        employees_data = employees_ref.get()
        if employees_data:
            for emp_id, emp_info in employees_data.items():
                folder_path = os.path.join(dataset_path, emp_id)
                if os.path.exists(folder_path):
                    images_count = len([
                        f for f in os.listdir(folder_path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                    ])
                    if 'images_count' not in emp_info or emp_info['images_count'] == 0:
                        emp_ref = db.reference(f'employees/{emp_id}')
                        emp_ref.update({'images_count': images_count})

                combined_data.append({
                    'id': emp_id,
                    'name': emp_info.get('name', 'Unknown'),
                    'golongan': '-',  # kosongkan karena karyawan
                    'semester': '-',  # kosongkan karena karyawan
                    'jabatan': emp_info.get('jabatan', 'Unknown'),
                    'images_count': emp_info.get('images_count', 0),
                    'edit_url': f'/students/edit/{emp_id}',  # kamu bisa sesuaikan endpoint edit-nya
                    'delete_url': f'/students/delete/{emp_id}'  # kamu bisa sesuaikan juga
                })

        return jsonify({'status': 'success', 'data': combined_data})

    except Exception as e:
        print(f"Error saat mengambil data: {str(e)}")
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


@app.route('/admin/edit_jadwal', methods=['POST'])
def edit_jadwal():
    """
    Route untuk mengedit jadwal mata kuliah.
    """
    if 'user' not in session:
        return redirect('/')

    try:
        # Ambil data dari form
        golongan_lama = request.form.get('golongan_lama')
        mata_kuliah_id = request.form.get('mata_kuliah_id')  # ID jadwal lama
        golongan_baru = request.form.get('golongan')  # Golongan baru
        kode_mk = request.form.get('kode_mk')  # Kode MK baru
        jumlah_pertemuan = int(request.form.get('jumlah_pertemuan', 0))  # Jumlah pertemuan baru
        mata_kuliah_baru = request.form.get('mata_kuliah')  # Nama mata kuliah baru
        start_time = request.form.get('start_time')  # Jam mulai baru
        end_time = request.form.get('end_time')  # Jam selesai baru

        # Validasi input
        if not all([golongan_lama, mata_kuliah_id, golongan_baru, kode_mk, jumlah_pertemuan, mata_kuliah_baru, start_time, end_time]):
            return redirect('/admin/jadwal_mata_kuliah?error=Semua field harus diisi!')

        # Referensi jadwal lama
        jadwal_ref_lama = db.reference(f'jadwal_mata_kuliah/{golongan_lama}/{mata_kuliah_id}')
        jadwal_data = jadwal_ref_lama.get()

        if jadwal_data:
            # Jika golongan berubah, pindahkan data ke golongan baru
            if golongan_lama != golongan_baru:
                jadwal_ref_lama.delete()  # Hapus data lama
                jadwal_ref_baru = db.reference(f'jadwal_mata_kuliah/{golongan_baru}/{mata_kuliah_id}')
                jadwal_ref_baru.set({
                    'kode_mk': kode_mk,
                    'name': mata_kuliah_baru,
                    'jumlah_pertemuan': jumlah_pertemuan,
                    'start_time': start_time,
                    'end_time': end_time
                })
            else:
                # Update data di lokasi yang sama
                jadwal_ref_lama.update({
                    'kode_mk': kode_mk,
                    'name': mata_kuliah_baru,
                    'jumlah_pertemuan': jumlah_pertemuan,
                    'start_time': start_time,
                    'end_time': end_time
                })
            message = "Jadwal berhasil diperbarui!"
        else:
            message = "Data jadwal lama tidak ditemukan."
    except Exception as e:
        message = f"Terjadi kesalahan saat memperbarui jadwal: {str(e)}"
        print(f"[ERROR] {message}")

    return redirect(f'/admin/jadwal_mata_kuliah?message={message}')


# Admin Melihat dan Mengelola Jadwal Kerja
@app.route('/admin/jadwal_kerja', methods=['GET', 'POST'])
def admin_jadwal_kerja():
    if 'user' not in session:
        return redirect('/')

    # Referensi ke Firebase Realtime Database
    jadwal_ref = db.reference('jadwal_kerja')
    edit_data = None
    message = None

    if request.method == 'POST':
        action = request.form.get('action')

        # Tambah jadwal baru
        if action == 'add':
            jabatan = request.form.get('jabatan')
            id_jadwal = request.form.get('id_jadwal')
            jam_masuk = request.form.get('jam_masuk')
            jam_pulang = request.form.get('jam_pulang')
            toleransi = request.form.get('toleransi', 15)

            if not all([jabatan, id_jadwal, jam_masuk, jam_pulang]):
                message = "Semua field wajib diisi!"
            else:
                try:
                    toleransi = int(toleransi)
                    jadwal_ref.child(jabatan).child(id_jadwal).set({
                        'id_jadwal': id_jadwal,
                        'jam_masuk': jam_masuk,
                        'jam_pulang': jam_pulang,
                        'toleransi_keterlambatan': toleransi
                    })
                    message = "Jadwal kerja berhasil ditambahkan!"
                except ValueError:
                    message = "Toleransi harus berupa angka!"

        # Persiapan edit jadwal
        elif action == 'prepare_edit':
            jabatan = request.form.get('edit_jabatan')
            id_jadwal = request.form.get('edit_id')

            if jabatan and id_jadwal:
                edit_ref = jadwal_ref.child(jabatan).child(id_jadwal)
                data = edit_ref.get()
                if data:
                    edit_data = {
                        'jabatan': jabatan,
                        'id_jadwal': id_jadwal,
                        'jam_masuk': data.get('jam_masuk', ''),
                        'jam_pulang': data.get('jam_pulang', ''),
                        'toleransi_keterlambatan': data.get('toleransi_keterlambatan', 15)
                    }
                else:
                    message = "Data tidak ditemukan!"
            else:
                message = "Data tidak lengkap untuk proses edit!"

        # Update data jadwal
        elif action == 'update':
            jabatan = request.form.get('jabatan')
            id_jadwal = request.form.get('id_jadwal')
            jam_masuk = request.form.get('jam_masuk')
            jam_pulang = request.form.get('jam_pulang')
            toleransi = request.form.get('toleransi', 15)

            if not all([jabatan, id_jadwal, jam_masuk, jam_pulang]):
                message = "Semua field wajib diisi!"
            else:
                try:
                    toleransi = int(toleransi)
                    jadwal_ref.child(jabatan).child(id_jadwal).update({
                        'jam_masuk': jam_masuk,
                        'jam_pulang': jam_pulang,
                        'toleransi_keterlambatan': toleransi
                    })
                    message = "Jadwal berhasil diperbarui!"
                except ValueError:
                    message = "Toleransi harus berupa angka!"

    # Ambil semua data jadwal dari Firebase
    jadwal_data = jadwal_ref.get()

    if jadwal_data is None:
        jadwal_data = {}

    data_list = []
    for jabatan, jadwal_list in jadwal_data.items():
        if isinstance(jadwal_list, dict):  # <- mencegah error jika bukan dict
            for id_jadwal, jadwal in jadwal_list.items():
                data_list.append({
                    'jabatan': jabatan,
                    'id_jadwal': jadwal.get('id_jadwal'),
                    'jam_masuk': jadwal.get('jam_masuk'),
                    'jam_pulang': jadwal.get('jam_pulang'),
                    'toleransi_keterlambatan': jadwal.get('toleransi_keterlambatan')
                })


    return render_template(
        'admin_jadwal_kerja.html',
        data_list=data_list,
        message=message,
        edit_data=edit_data
    )


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
    
@app.route('/absen', methods=['GET', 'POST'])
def absen():
    if 'karyawan' not in session:
        flash('Silakan login terlebih dahulu', 'warning')
        return redirect('/login_karyawan')

    karyawan = session['karyawan']
    jabatan = karyawan.get('jabatan', '')

    jadwal_kerja_data = []
    jadwal_ref = db.reference(f'jadwal_kerja/{jabatan}')
    jadwal_data = jadwal_ref.get()

    if jadwal_data:
        for k, v in jadwal_data.items():
            jadwal_kerja = {
                'id': k,
                'name': v.get('name', 'Tidak Ada Nama'),
                'jam_masuk': v.get('jam_masuk', ''),
                'jam_pulang': v.get('jam_pulang', ''),
                'status': v.get('status', 'Tidak Hadir'),
                'is_terlambat': False
            }
            
            # Periksa apakah waktu absensi sudah lewat
            jam_masuk = jadwal_kerja['jam_masuk']
            if jam_masuk:
                jam_masuk_time = datetime.strptime(jam_masuk, '%H:%M')
                now = datetime.now()

                if now > jam_masuk_time and now - jam_masuk_time > timedelta(minutes=15):
                    jadwal_kerja['is_terlambat'] = True
                else:
                    jadwal_kerja['is_terlambat'] = False

            jadwal_kerja_data.append(jadwal_kerja)

    return render_template('absen.html', karyawan=karyawan, jadwal_kerja_data=jadwal_kerja_data)


def run_face_recognition(user_id):
    """
    Fungsi untuk deteksi wajah dan menyimpan data absensi.
    """
    username_full = labels.get(user_id, "Unknown")
    username = username_full.split("", 1)[1] if "" in username_full else username_full

    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("[ERROR] Kamera tidak dapat dibuka. Pastikan kamera tersedia.")
        return

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    locked_label = "Unknown"
    lock_frames = 0
    threshold_confidence = 0.75
    min_consecutive_frames = 5
    attendance_logged = False

    print("[INFO] Mulai deteksi wajah...")
    while True:
        ret, frame = video.read()
        if not ret:
            print("[ERROR] Tidak dapat membaca frame dari kamera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            print("[INFO] Tidak ada wajah yang terdeteksi.")
        else:
            print(f"[INFO] {len(faces)} wajah terdeteksi.")

        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (224, 224))
            face_img = np.expand_dims(face_img, axis=0) / 255.0

            prediction = model.predict(face_img)
            confidence = float(np.max(prediction[0]))
            id_detected = str(np.argmax(prediction[0]) + 1)

            print(f"[DEBUG] Deteksi: ID={id_detected}, Confidence={confidence:.2f}, Lock frames={lock_frames}")

            if confidence >= threshold_confidence and id_detected == user_id:
                lock_frames += 1
                if lock_frames >= min_consecutive_frames and not attendance_logged:
                    try:
                        # Simpan frame sebagai gambar
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_path = f"attendance_{user_id}_{timestamp}.jpg"
                        cv2.imwrite(image_path, frame)
                        print("[INFO] Gambar berhasil disimpan:", image_path)

                        # Unggah gambar ke Firebase Storage
                        try:
                            blob = bucket.blob(f'attendance_images/{image_path}')
                            blob.upload_from_filename(image_path)
                            blob.make_public()
                            image_url = blob.public_url
                            print("[SUCCESS] Gambar berhasil diunggah ke Firebase Storage. URL:", image_url)
                        except Exception as e:
                            print("[ERROR] Gagal mengunggah gambar ke Firebase Storage:", str(e))
                            continue

                        # Simpan data ke Firebase Realtime Database
                        try:
                            attendance_data = {
                                'user_id': user_id,
                                'username': username,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'confidence': confidence,
                                'image_url': image_url
                            }
                            db.reference(f'attendance/{user_id}').push(attendance_data)
                            print("[SUCCESS] Data presensi berhasil disimpan:", attendance_data)
                        except Exception as e:
                            print("[ERROR] Gagal menyimpan data ke Firebase Realtime Database:", str(e))
                            continue

                        # Hapus gambar lokal
                        os.remove(image_path)
                        print("[INFO] Gambar lokal dihapus:", image_path)

                        attendance_logged = True  # Tandai presensi sudah tercatat
                        break  # Keluar dari loop setelah presensi berhasil
                    except Exception as e:
                        print("[ERROR] Terjadi kesalahan saat mencatat presensi:", str(e))
            else:
                lock_frames = 0

        if attendance_logged:
                 print("[INFO] Presensi selesai, keluar dari loop.")
                 break

    video.release()
    cv2.destroyAllWindows()
    if not attendance_logged:
        print("[ERROR] Presensi tidak tercatat. Pastikan wajah terlihat jelas.")

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


@app.route('/check_absen_status', methods=['GET'])
def check_absen_status():
    # Mendapatkan parameter dari URL
    user_id = request.args.get('user_id')
    jadwal_kerja = request.args.get('jadwal_kerja')  # Harus berupa kode asli
    minggu_ke = request.args.get('minggu_ke')

    # Validasi parameter
    if not user_id or not jadwal_kerja or not minggu_ke:
        print("[ERROR] Parameter tidak lengkap. Pastikan 'user_id', 'jadwal_kerja', dan 'minggu_ke' disertakan.")
        return jsonify({"status": "error", "message": "Parameter tidak lengkap"}), 400

    print(f"[DEBUG] Checking attendance for user_id={user_id}, mata_kuliah={jadwal_kerja}, minggu_ke={minggu_ke}")

    try:
        # Referensi ke lokasi data absensi di Firebase Realtime Database
        attendance_ref = db.reference(f"attendance/{jadwal_kerja}/{minggu_ke}/{user_id}")
        data = attendance_ref.get()

        if data:
            print("[DEBUG] Attendance Data Found:", data)
            return jsonify({"status": "success", "data": data})
        else:
            print("[DEBUG] Attendance Data Not Found")
            return jsonify({"status": "pending", "message": "Data absensi tidak ditemukan"}), 404
    except Exception as e:
        # Menangkap kesalahan selama proses membaca data dari Firebase
        print(f"[ERROR] Terjadi kesalahan saat memeriksa status absensi: {e}")
        return jsonify({"status": "error", "message": "Terjadi kesalahan server"}), 500


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
            dataset_path = "DataSet"
            test_dataset_path = "DataTest"
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
    app.run(debug=True, port=5000)