-- Create Database
CREATE DATABASE IF NOT EXISTS pro17;
USE pro17;

-- Table: admin
CREATE TABLE IF NOT EXISTS admin (
    id INT AUTO_INCREMENT PRIMARY KEY,
    a_username VARCHAR(255) NOT NULL,
    a_password VARCHAR(255) NOT NULL
);

-- Insert default admin user
INSERT INTO admin (a_username, a_password) VALUES ('admin', 'admin');

-- Table: user
CREATE TABLE IF NOT EXISTS user (
    u_id INT AUTO_INCREMENT PRIMARY KEY,
    u_name VARCHAR(255) NOT NULL,
    u_email VARCHAR(255) NOT NULL UNIQUE,
    u_password VARCHAR(255) NOT NULL,
    u_mobile VARCHAR(20),
    u_age INT
);

-- Insert dummy users
INSERT INTO user (u_name, u_email, u_password, u_mobile, u_age) VALUES 
('John Doe', 'john@example.com', 'password123', '1234567890', 30),
('Jane Smith', 'jane@example.com', 'password456', '0987654321', 25);

-- Table: doctors
CREATE TABLE IF NOT EXISTS doctors (
    d_id INT AUTO_INCREMENT PRIMARY KEY,
    d_name VARCHAR(255) NOT NULL,
    d_email VARCHAR(255) NOT NULL UNIQUE,
    d_passwords VARCHAR(255) NOT NULL,
    d_spec VARCHAR(255) NOT NULL
);

-- Insert dummy doctors
INSERT INTO doctors (d_name, d_email, d_passwords, d_spec) VALUES 
('Dr. Alice', 'alice@hospital.com', 'doctor123', 'Cardiologist'),
('Dr. Bob', 'bob@hospital.com', 'doctor456', 'Dermatologist'),
('Dr. Charlie', 'charlie@hospital.com', 'doctor789', 'Neurologist');

-- Table: appointment
CREATE TABLE IF NOT EXISTS appointment (
    ap_id INT AUTO_INCREMENT PRIMARY KEY,
    d_id INT,
    u_id INT,
    ap_time VARCHAR(50),
    ap_date VARCHAR(50),
    ap_report VARCHAR(255),
    ap_payment_status VARCHAR(50) DEFAULT 'pending',
    ap_status VARCHAR(50) DEFAULT 'pending',
    razorpay_order_id VARCHAR(255),
    FOREIGN KEY (d_id) REFERENCES doctors(d_id) ON DELETE CASCADE,
    FOREIGN KEY (u_id) REFERENCES user(u_id) ON DELETE CASCADE
);

-- Insert dummy appointments
INSERT INTO appointment (d_id, u_id, ap_time, ap_date, ap_report, ap_payment_status, ap_status, razorpay_order_id) VALUES 
(1, 1, '10:00 AM', '2023-10-25', 'report1.pdf', 'success', 'Accept', 'order_12345'),
(2, 2, '02:00 PM', '2023-10-26', 'report2.pdf', 'pending', 'pending', 'order_67890');
