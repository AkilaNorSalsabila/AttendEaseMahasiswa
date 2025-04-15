<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Dashboard</title>

    <!-- Custom fonts for this template -->
    <link href="/static/vendor/fontawesome-free/css/all.min.css" rel="stylesheet" type="text/css">
    <link href="https://fonts.googleapis.com/css?family=Nunito:200,200i,300,300i,400,400i,600,600i,700,700i,800,800i,900,900i" rel="stylesheet">

    <!-- Custom styles for this template -->
    <link href="/static/css/sb-admin-2.min.css?v={{ timestamp }}" rel="stylesheet">



    <script>
        function loadStudents() {
            const loader = `<tr><td colspan="7" style="text-align: center;">Memuat data...</td></tr>`;
            const tableBody = document.getElementById('students-table-body');
            tableBody.innerHTML = loader;

            fetch('/students')
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        tableBody.innerHTML = ''; // Kosongkan tabel sebelumnya

                        if (data.data.length === 0) {
                            tableBody.innerHTML = `<tr><td colspan="7" style="text-align: center;">Tidak ada data mahasiswa.</td></tr>`;
                        } else {
                            data.data.forEach(student => {
                                const row = `
                                    <tr>
                                        <td>${student.id}</td>
                                        <td>${student.name}</td>
                                        <td>${student.nim}</td>
                                        <td>${student.images_count}</td>
                                        <td>${student.semester || 'Tidak Ada'}</td>
                                        <td>${student.golongan || 'Tidak Ada'}</td>
                                        <td>
                                            <button class="btn btn-warning btn-sm" onclick="editStudent('${student.id}', '${student.name}', '${student.nim}', '${student.semester}', '${student.golongan}')">Edit</button>
                                            <button class="btn btn-danger btn-sm" onclick="deleteStudent('${student.id}')">Hapus</button>
                                        </td>
                                    </tr>`;
                                tableBody.innerHTML += row;
                            });
                        }
                    } else {
                        console.error("Gagal memuat data mahasiswa:", data.message);
                        tableBody.innerHTML =
                            '<tr><td colspan="7" style="text-align: center;">Gagal memuat data mahasiswa.</td></tr>';
                    }
                })
                .catch(err => {
                    console.error("Error saat memuat data mahasiswa:", err.message);
                    tableBody.innerHTML =
                        '<tr><td colspan="7" style="text-align: center;">Terjadi kesalahan jaringan.</td></tr>';
                });
        }

        function editStudent(studentId, currentName, currentNim, currentSemester, currentGolongan) {
        const newGolongan = prompt('Masukkan golongan baru:', currentGolongan);
        const newSemester = prompt('Masukkan semester baru:', currentSemester);

        if (newGolongan && newSemester) {
            fetch(`/students/edit/${studentId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    golongan: newGolongan,
                    semester: newSemester
                })
            })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        alert(data.message);
                        loadStudents(); // Reload data setelah edit berhasil
                    } else {
                        alert(`Gagal mengedit mahasiswa: ${data.message}`);
                    }
                })
                .catch(err => {
                    console.error("Error saat mengedit mahasiswa:", err.message);
                    alert('Terjadi kesalahan saat mengedit data mahasiswa.');
                });
        } else {
            alert("Golongan dan Semester tidak boleh kosong.");
        }
    }

    function deleteStudent(studentId) {
        if (confirm('Yakin ingin menghapus mahasiswa ini?')) {
            fetch(`/students/delete/${studentId}`, { method: 'DELETE' })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        alert(data.message);
                        loadStudents(); // Reload data setelah delete berhasil
                    } else {
                        alert(`Gagal menghapus mahasiswa: ${data.message}`);
                    }
                })
                .catch(err => {
                    console.error("Error saat menghapus mahasiswa:", err.message);
                    alert('Terjadi kesalahan saat menghapus data mahasiswa.');
                });
        }
    }
        window.onload = loadStudents;
    </script>
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
                            <a class="nav-link dropdown-toggle" href="#" id="userDropdown" role="button" data-toggle="dropdown">
                                <span class="mr-2 d-none d-lg-inline text-gray-600 small">Admin</span>
                                <img class="img-profile rounded-circle" src="/static/img/undraw_profile.svg">
                            </a>
                        </li>
                    </ul>
                </nav>
                <!-- End of Topbar -->

                <!-- Begin Page Content -->
                <div class="container-fluid">

                    <!-- Page Heading -->
                    <h1 class="h3 mb-4 text-gray-800">Welcome, Admin!</h1>

                    <!-- Table Section -->
                    <div class="row">
                        <div class="col-lg-12">
                            <div class="card shadow mb-4">
                                <div class="card-header py-3">
                                    <h6 class="m-0 font-weight-bold text-primary">Daftar Mahasiswa</h6>
                                </div>
                                <div class="card-body">
                                    <div class="table-responsive">
                                        <table class="table table-bordered" width="100%" cellspacing="0">
                                            <thead>
                                                <tr>
                                                    <th>ID Mahasiswa</th>
                                                    <th>Nama</th>
                                                    <th>NIM</th>
                                                    <th>Jumlah Gambar</th>
                                                    <th>Semester</th>
                                                    <th>Golongan</th>
                                                    <th>Aksi</th>
                                                </tr>
                                            </thead>
                                            <tbody id="students-table-body">
                                                <tr>
                                                    <td colspan="7" style="text-align: center;">Memuat data...</td>
                                                </tr>
                                            </tbody>    
                                        </table>
                                    </div>
                                </div>
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

    <!-- Core plugin JavaScript -->
    <script src="/static/vendor/jquery-easing/jquery.easing.min.js"></script>

    <!-- Custom scripts for all pages -->
    <script src="/static/js/sb-admin-2.min.js"></script>

</body>

</html>
