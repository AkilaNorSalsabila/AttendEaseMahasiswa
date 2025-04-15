<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Absensi</title>

    <!-- Custom fonts for this template -->
    <link href="/static/vendor/fontawesome-free/css/all.min.css" rel="stylesheet" type="text/css">
    <link href="https://fonts.googleapis.com/css?family=Nunito:200,200i,300,300i,400,400i,600,600i,700,700i,800,800i,900,900i" rel="stylesheet">

    <!-- Custom styles for this template -->
    <link href="/static/css/sb-admin-2.min.css" rel="stylesheet">

    <!-- Bootstrap core CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">

    <style>
        .modal-header, .modal-body, .modal-footer {
            padding: 15px;
        }
    </style>
</head>

<body id="page-top">

    <!-- Page Wrapper -->
    <div id="wrapper">

        <!-- Sidebar -->
        <ul class="navbar-nav bg-gradient-primary sidebar sidebar-dark accordion" id="accordionSidebar">
            <a class="sidebar-brand d-flex align-items-center justify-content-center" href="/dashboard">
                <div class="sidebar-brand-icon rotate-n-15">
                    <i class="fas fa-laugh-wink"></i>
                </div>
                <div class="sidebar-brand-text mx-3">Admin Dashboard</div>
            </a>
            <hr class="sidebar-divider">
            <li class="nav-item">
                <a class="nav-link" href="/dashboard">
                    <i class="fas fa-database"></i>
                    <span>Dashboard</span>
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="/dataset">
                    <i class="fas fa-database"></i>
                    <span>Tambah Dataset</span>
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="/train">
                    <i class="fas fa-brain"></i>
                    <span>Training Model</span>
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="/attendance">
                    <i class="fas fa-list"></i>
                    <span>Mahasiswa Attendance</span>
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="/admin/jadwal_mata_kuliah">
                    <i class="fas fa-calendar"></i>
                    <span>Jadwal Mata Kuliah</span>
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link logout" href="/logout">
                    <i class="fas fa-sign-out-alt"></i>
                    <span>Logout</span>
                </a>
            </li>
        </ul>

        <!-- Content Wrapper -->
        <div id="content-wrapper" class="d-flex flex-column">

            <!-- Main Content -->
            <div id="content">

                <!-- Topbar -->
                <nav class="navbar navbar-expand navbar-light bg-white topbar mb-4 static-top shadow">
                    <!-- Topbar content here... -->
                </nav>

                <!-- Begin Page Content -->
                <div class="container-fluid">

                    <h1 class="h3 mb-4 text-gray-800 text-center">Data Kehadiran Mahasiswa</h1>

                    <!-- Form Pilih Golongan -->
                    <form method="POST" action="/attendance">
                        <div class="mb-3">
                            <label for="golongan">Pilih Golongan:</label>
                            <select name="golongan" id="golongan" class="form-control" onchange="this.form.submit()">
                                <option value="">Pilih Golongan</option>
                                {% for golongan in golongan_list %}
                                    <option value="{{ golongan }}" {% if golongan == request.form['golongan'] %} selected {% endif %}>{{ golongan }}</option>
                                {% endfor %}
                            </select>
                        </div>
                    </form>
                    
                    {% if golongan %}
                    <!-- Form untuk memilih mata kuliah -->
                    <form method="POST" action="/attendance">
                        <input type="hidden" name="golongan" value="{{ golongan }}">
                        <div class="mb-3">
                            <label for="mata_kuliah">Pilih Mata Kuliah:</label>
                            <select name="mata_kuliah" id="mata_kuliah" class="form-control" onchange="this.form.submit()">
                                <option value="">Pilih Mata Kuliah</option>
                                {% for mata_kuliah in mata_kuliah_list %}
                                    <option value="{{ mata_kuliah }}" {% if mata_kuliah == request.form['mata_kuliah'] %} selected {% endif %}>{{ mata_kuliah }}</option>
                                {% endfor %}
                            </select>
                        </div>
                    </form>
                    {% endif %}

                    {% if golongan and mata_kuliah %}
                    <h3 class="mt-4 text-center">Rekap Absensi - Mata Kuliah: {{ mata_kuliah }}</h3>
                    <table class="table table-bordered table-striped">
                        <thead>
                            <tr>
                                <th>Kode Mata Kuliah</th>
                                <th>Nama Mata Kuliah</th>
                                <th>Minggu Ke</th>
                                <th>NIM</th>
                                <th>Nama</th>
                                <th>Status</th>
                                <th>Waktu Kehadiran</th>
                                <th>Bukti Kehadiran</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for attendance in attendance_list %}
                                <tr>
                                    <td>{{ attendance.kode_mata_kuliah }}</td>
                                    <td>{{ attendance.nama_mata_kuliah }}</td>
                                    <td>{{ attendance.minggu_ke }}</td>
                                    <td>{{ attendance.nim }}</td>
                                    <td>{{ attendance.nama }}</td>
                                    <td>{{ attendance.status }}</td>
                                    <td>{{ attendance.timestamp }}</td>
                                    <td>
                                        {% if attendance.image_url %}
                                            <img src="{{ attendance.image_url }}" alt="Bukti Kehadiran" width="50" height="50">
                                        {% else %}
                                            Tidak Ada
                                        {% endif %}
                                    </td>
                                </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>

    <script src="/static/vendor/jquery/jquery.min.js"></script>
    <script src="/static/vendor/bootstrap/js/bootstrap.bundle.min.js"></script>
</body>

</html>
