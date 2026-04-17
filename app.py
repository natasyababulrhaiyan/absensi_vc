from flask import Flask, render_template, request, redirect, session
import mysql.connector

app = Flask(__name__)
app.secret_key = "secret123"

# ================= KONEKSI DATABASE =================
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",  # default laragon kosong
    database="db_absensi"
)

# ================= LOGIN =================
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        nim = request.form['nim']
        password = request.form['password']

        cursor = db.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM users WHERE nim=%s AND password=%s",
            (nim, password)
        )
        user = cursor.fetchone()

        if user:
            session['nama'] = user['nama']
            session['role'] = user['role']

            if user['role'] == 'admin':
                return redirect('/admin')
            else:
                return redirect('/peserta')
        else:
            return "Login gagal!"

    return render_template('login.html')


# ================= HALAMAN ADMIN =================
@app.route('/admin')
def admin():
    if 'role' not in session or session['role'] != 'admin':
        return "Akses ditolak!"

    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM absensi ORDER BY waktu_masuk DESC")
    data = cursor.fetchall()

    return render_template('admin.html', data=data)


# ================= HALAMAN PESERTA =================
@app.route('/peserta')
def peserta():
    if 'role' not in session or session['role'] != 'peserta':
        return "Akses ditolak!"

    return render_template('peserta.html', nama=session['nama'])


# ================= ABSENSI =================
@app.route('/absen', methods=['POST'])
def absen():
    if 'role' not in session or session['role'] != 'peserta':
        return "Akses ditolak!"

    nim = request.form.get('nim')
    nama = session['nama']
    status = request.form.get('status')
    keterangan = request.form.get('keterangan', 'Terverifikasi')

    cursor = db.cursor()

    cursor.execute("""
        INSERT INTO absensi (nim, nama, status, keterangan)
        VALUES (%s, %s, %s, %s)
    """, (nim, nama, status, keterangan))

    db.commit()

    return "OK"

# ================= HALAMAN DATA ABSENSI =================
@app.route('/data-absensi', methods=['GET', 'POST'])
def data_absensi():
    if 'role' not in session or session['role'] != 'admin':
        return "Akses ditolak!"

    cursor = db.cursor(dictionary=True)

    if request.method == 'POST':
        tanggal = request.form.get('tanggal')

        cursor.execute("""
            SELECT * FROM absensi 
            WHERE DATE(waktu_masuk) = %s
            ORDER BY waktu_masuk DESC
        """, (tanggal,))
    else:
        cursor.execute("SELECT * FROM absensi ORDER BY waktu_masuk DESC")

    data = cursor.fetchall()

    return render_template('data_absensi.html', data=data)

# ================= LOGOUT =================
@app.route('/keluar', methods=['POST'])
def keluar():
    nim = request.form.get('nim')

    cursor = db.cursor()
    cursor.execute("""
        UPDATE absensi 
        SET waktu_keluar = CURRENT_TIMESTAMP
        WHERE nim = %s AND waktu_keluar IS NULL
    """, (nim,))

    db.commit()
    return "OK"

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

# ================= RUN APP =================
if __name__ == '__main__':
    app.run(debug=True)