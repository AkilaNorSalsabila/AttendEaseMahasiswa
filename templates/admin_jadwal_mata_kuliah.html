<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Admin panel untuk mengelola jadwal mata kuliah.">
    <title>Jadwal Mata Kuliah</title>

    <!-- Custom fonts for this template -->
    <link href="/static/vendor/fontawesome-free/css/all.min.css" rel="stylesheet" type="text/css">
    <link href="https://fonts.googleapis.com/css?family=Nunito:200,200i,300,300i,400,400i,600,600i,700,700i,800,800i,900,900i" rel="stylesheet">

    <!-- Custom styles for this template -->
    <link href="/static/css/sb-admin-2.min.css" rel="stylesheet">
</head>

<body id="page-top">

    <!-- Page Wrapper -->
    <div id="wrapper">

        <!-- Sidebar -->
        <ul class="navbar-nav bg-gradient-primary sidebar sidebar-dark accordion" id="accordionSidebar">

            <!-- Sidebar - Brand -->
            <a class="sidebar-brand d-flex align-items-center justify-content-center" href="/dashboard">
                <div class="sidebar-brand-icon rotate-n-15">
                    <i class="fas fa-laugh-wink"></i>
                </div>
                <div class="sidebar-brand-text mx-3">Admin Dashboard</div>
            </a>

            <!-- Divider -->
            <hr class="sidebar-divider">

            <!-- Sidebar Links -->
            <li class="nav-item">
                <a class="nav-link" href="/dashboard">
                    <i class="fas fa-tachometer-alt"></i>
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
        <!-- End of Sidebar -->

        <!-- Content Wrapper -->
        <div id="content-wrapper" class="d-flex flex-column">

            <!-- Main Content -->
            <div id="content">

                <!-- Topbar -->
                <nav class="navbar navbar-expand navbar-light bg-white topbar mb-4 static-top shadow">
                    <button id="sidebarToggleTop" class="btn btn-link d-md-none rounded-circle mr-3">
                        <i class="fa fa-bars"></i>
                    </button>

                    <!-- Topbar Navbar -->
                    <ul class="navbar-nav ml-auto">
                        <li class="nav-item dropdown no-arrow">
                            <a class="nav-link dropdown-toggle" href="#" id="userDropdown" role="button">
                                <span class="mr-2 d-none d-lg-inline text-gray-600 small">Admin</span>
                                <img class="img-profile rounded-circle" src="/static/img/undraw_profile.svg" alt="Profile Image">
                            </a>
                        </li>
                    </ul>
                </nav>
                <!-- End of Topbar -->

                <!-- Begin Page Content -->
                <div class="container-fluid">

                    <!-- Page Heading -->
                    <h1 class="h3 mb-4 text-gray-800 text-center">Jadwal Mata Kuliah</h1>

                    <!-- Form Tambah/Edit Jadwal -->
                    <div class="card shadow mb-4">
                        <div class="card-body">
                            <form method="POST" action="/admin/jadwal_mata_kuliah" class="user">
                                <input type="hidden" name="action" value="{{ 'update' if edit_data else 'add' }}">
                                {% if edit_data %}
                                <input type="hidden" name="edit_golongan" value="{{ edit_data.golongan }}">
                                <input type="hidden" name="edit_id" value="{{ edit_data.id }}">
                                {% endif %}
                                <div class="form-group">
                                    <label for="golongan">Golongan:</label>
                                    <select id="golongan" name="golongan" class="form-control" {% if edit_data %}disabled{% endif %} required>
                                        <option value="">Pilih Golongan</option>
                                        {% for golongan in dropdown_golongan %}
                                        <option value="{{ golongan }}" {% if edit_data and edit_data.golongan == golongan %}selected{% endif %}>
                                            {{ golongan }}
                                        </option>
                                        {% endfor %}
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="kode_mk">Kode Mata Kuliah:</label>
                                    <input type="text" id="kode_mk" name="kode_mk" class="form-control" value="{{ edit_data.kode_mk if edit_data else '' }}" required>
                                </div>
                                <div class="form-group">
                                    <label for="mata_kuliah">Mata Kuliah:</label>
                                    <input type="text" id="mata_kuliah" name="mata_kuliah" class="form-control" value="{{ edit_data.name if edit_data else '' }}" required>
                                </div>
                                <div class="form-group">
                                    <label for="jumlah_pertemuan">Jumlah Pertemuan:</label>
                                    <input type="number" id="jumlah_pertemuan" name="jumlah_pertemuan" class="form-control" value="{{ edit_data.jumlah_pertemuan if edit_data else '' }}" required>
                                </div>
                                <div class="form-group">
                                    <label for="start_time">Jam Mulai:</label>
                                    <input type="time" id="start_time" name="start_time" class="form-control" value="{{ edit_data.start_time if edit_data else '' }}" required>
                                </div>
                                <div class="form-group">
                                    <label for="end_time">Jam Selesai:</label>
                                    <input type="time" id="end_time" name="end_time" class="form-control" value="{{ edit_data.end_time if edit_data else '' }}" required>
                                </div>
                                <button type="submit" class="btn btn-primary btn-user btn-block">
                                    {{ 'Simpan Perubahan' if edit_data else 'Tambah Jadwal' }}
                                </button>
                            </form>
                        </div>
                    </div>

                    <!-- Daftar Jadwal -->
                    <div class="card shadow mb-4">
                        <div class="card-body">
                            <h5 class="text-gray-800 mb-3">Daftar Jadwal Mata Kuliah</h5>
                            <div class="table-responsive">
                                <table class="table table-bordered">
                                    <thead class="thead-light">
                                        <tr>
                                            <th>Golongan</th>
                                            <th>Kode MK</th>
                                            <th>Mata Kuliah</th>
                                            <th>Jumlah Pertemuan</th>
                                            <th>Jam Mulai</th>
                                            <th>Jam Selesai</th>
                                            <th>Aksi</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {% for jadwal in golongan_data %}
                                        <tr>
                                            <td>{{ jadwal.golongan }}</td>
                                            <td>{{ jadwal.kode_mk }}</td>
                                            <td>{{ jadwal.mata_kuliah }}</td>
                                            <td>{{ jadwal.jumlah_pertemuan }}</td>
                                            <td>{{ jadwal.start_time }}</td>
                                            <td>{{ jadwal.end_time }}</td>
                                            <td>
                                                <form method="POST" action="/admin/jadwal_mata_kuliah" style="display:inline;">
                                                    <input type="hidden" name="action" value="prepare_edit">
                                                    <input type="hidden" name="edit_golongan" value="{{ jadwal.golongan }}">
                                                    <input type="hidden" name="edit_id" value="{{ jadwal.id }}">
                                                    <button type="submit" class="btn btn-warning btn-sm" title="Edit Jadwal">Edit
                                                        <i class="fas fa-pencil-alt"></i>
                                                    </button>
                                                </form>
                                                <form method="POST" action="/admin/jadwal_mata_kuliah/delete/{{ jadwal.golongan }}/{{ jadwal.id }}" style="display:inline;">
                                                    <button type="submit" class="btn btn-danger btn-sm" title="Hapus Jadwal" onclick="return confirm('Yakin ingin menghapus jadwal ini?')">Hapus
                                                        <i class="fas fa-trash-alt"></i>
                                                    </button>
                                                </form>
                                            </td>
                                        </tr>
                                        {% endfor %}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>

                </div>
                <!-- /.container-fluid -->

            </div>
            <!-- End of Main Content -->


        </div>
        <!-- End of Content Wrapper -->

    </div>
    <!-- End of Page Wrapper -->

    <!-- Bootstrap core JavaScript -->
    <script src="/static/vendor/jquery/jquery.min.js"></script>
    <script src="/static/vendor/bootstrap/js/bootstrap.bundle.min.js"></script>
    <script src="/static/vendor/jquery-easing/jquery.easing.min.js"></script>
    <script src="/static/js/sb-admin-2.min.js"></script>

</body>

</html>