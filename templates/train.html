<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Model</title>

    <!-- Custom fonts for this template -->
    <link href="/static/vendor/fontawesome-free/css/all.min.css" rel="stylesheet" type="text/css">
    <link href="https://fonts.googleapis.com/css?family=Nunito:200,200i,300,300i,400,400i,600,600i,700,700i,800,800i,900,900i" rel="stylesheet">

    <!-- Custom styles for this template -->
    <link href="/static/css/sb-admin-2.min.css" rel="stylesheet">
    <style>
        /* Styling untuk container progress bar */
        .progress-container {
            display: flex;
            width: 100%;
            background-color: #f3f3f3;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin: 20px 0;
            align-items: center;  /* Menyelaraskan progress bar dan persentase secara vertikal */
        }

        /* Styling untuk progress bar */
        #w3-container {
            height: 30px;
            width: 0%;
            background-color: #4caf50;
            border-radius: 10px;
            text-align: center;
            color: white;
            line-height: 30px; /* Vertically center text */
        }

        /* Styling untuk persentase */
        #progress-text {
            font-weight: bold;
            color: #333;
            margin-left: 10px;
            line-height: 30px;
        }
    </style>
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
                    <h1 class="h3 mb-4 text-gray-800 text-center">Training Model</h1>

                    <!-- Training Button -->
                    <div class="row justify-content-center">
                        <div class="col-lg-6">
                            <div class="card shadow mb-4">
                                <div class="card-body text-center">
                                    <form method="POST" action="/train">
                                        <button type="submit" class="btn btn-success btn-lg" onclick="startTraining(event)">Mulai Training</button>
                                    </form>

                                    {% if message %}
                                        <p class="text-success mt-3">{{ message }}</p>
                                    {% endif %}
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Progress Bar -->
                    <div class="progress-container">
                        <div id="myBar" class="w3-container w3-green" style="width: 0%;"></div>
                        <!-- Menampilkan persentase di samping progress bar -->
                        <span id="progress-text" style="font-weight: bold; color: #333; margin-left: 10px; line-height: 30px;">0%</span>
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

    <!-- Progress Bar JavaScript -->
    <script>
        // Fungsi untuk mengupdate progress bar
        function updateProgressBar() {
            fetch('/get_progress')
                .then(response => response.json())
                .then(data => {
                    const progress = data.progress || 0; // Ambil progress dari JSON, default ke 0 jika tidak ada
                    const progressBar = document.getElementById('myBar');
                    const progressText = document.getElementById('progress-text');

                    progressBar.style.width = progress + '%';
                    progressText.innerText = progress + '%'; // Update persentase di samping

                    // Update progress setiap 2 detik
                    if (progress < 100) {
                        setTimeout(updateProgressBar, 2000); // Cek progres setiap 2 detik
                    } else {
                        document.getElementById('message').innerText = "Pelatihan Selesai!";
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('message').innerText = "Terjadi kesalahan saat memuat progres.";
                });
        }

        // Panggil fungsi updateProgressBar ketika halaman dimuat
        window.onload = updateProgressBar;
    </script>

</body>

</html>
