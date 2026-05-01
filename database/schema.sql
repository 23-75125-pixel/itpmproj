CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(50) PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    full_name VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('admin', 'user')),
    section_name VARCHAR(255) NOT NULL DEFAULT '',
    avatar_url TEXT NOT NULL DEFAULT '',
    password_hash TEXT NOT NULL,
    created_at DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS quizzes (
    id VARCHAR(50) PRIMARY KEY,
    creator_id VARCHAR(50) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    subject VARCHAR(255) NOT NULL,
    time_limit_minutes INT NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('draft', 'published', 'closed')),
    quiz_code VARCHAR(50) NOT NULL UNIQUE,
    monitoring_enabled BOOLEAN NOT NULL DEFAULT 0,
    assigned_section VARCHAR(255) NOT NULL DEFAULT '',
    scheduled_start DATETIME,
    scheduled_end DATETIME,
    created_at DATETIME NOT NULL,
    FOREIGN KEY (creator_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS questions (
    id VARCHAR(50) PRIMARY KEY,
    quiz_id VARCHAR(50) NOT NULL,
    question_text TEXT NOT NULL,
    question_type VARCHAR(30) NOT NULL CHECK (question_type IN ('multiple_choice', 'true_false', 'short_answer')),
    points INT NOT NULL DEFAULT 1,
    FOREIGN KEY (quiz_id) REFERENCES quizzes(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS question_options (
    id VARCHAR(50) PRIMARY KEY,
    question_id VARCHAR(50) NOT NULL,
    option_text TEXT NOT NULL,
    is_correct BOOLEAN NOT NULL DEFAULT 0,
    FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS quiz_attempts (
    id VARCHAR(50) PRIMARY KEY,
    quiz_id VARCHAR(50) NOT NULL,
    student_id VARCHAR(50) NOT NULL,
    quiz_code VARCHAR(50) NOT NULL,
    score INT NOT NULL DEFAULT 0,
    percentage DECIMAL(5,2) NOT NULL DEFAULT 0,
    status VARCHAR(20) NOT NULL CHECK (status IN ('in_progress', 'submitted', 'auto_submitted')),
    started_at DATETIME NOT NULL,
    submitted_at DATETIME,
    consent_given BOOLEAN NOT NULL DEFAULT 0,
    FOREIGN KEY (quiz_id) REFERENCES quizzes(id) ON DELETE CASCADE,
    FOREIGN KEY (student_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS student_responses (
    id VARCHAR(50) PRIMARY KEY,
    attempt_id VARCHAR(50) NOT NULL,
    question_id VARCHAR(50) NOT NULL,
    selected_option VARCHAR(255),
    text_response TEXT,
    is_correct BOOLEAN NOT NULL DEFAULT 0,
    FOREIGN KEY (attempt_id) REFERENCES quiz_attempts(id) ON DELETE CASCADE,
    FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS activity_logs (
    id VARCHAR(50) PRIMARY KEY,
    quiz_id VARCHAR(50) NOT NULL,
    attempt_id VARCHAR(50) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    event_description TEXT NOT NULL,
    flag_level VARCHAR(20) NOT NULL DEFAULT 'low' CHECK (flag_level IN ('low', 'medium', 'high')),
    reviewed BOOLEAN NOT NULL DEFAULT 0,
    instructor_notes TEXT,
    created_date DATETIME NOT NULL,
    FOREIGN KEY (quiz_id) REFERENCES quizzes(id) ON DELETE CASCADE,
    FOREIGN KEY (attempt_id) REFERENCES quiz_attempts(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_quizzes_creator_id ON quizzes(creator_id);
CREATE INDEX IF NOT EXISTS idx_quizzes_status_schedule ON quizzes(status, scheduled_start, scheduled_end);
CREATE INDEX IF NOT EXISTS idx_quizzes_assigned_section ON quizzes(assigned_section);
CREATE INDEX IF NOT EXISTS idx_questions_quiz_id ON questions(quiz_id);
CREATE INDEX IF NOT EXISTS idx_question_options_question_id ON question_options(question_id);
CREATE INDEX IF NOT EXISTS idx_quiz_attempts_quiz_id ON quiz_attempts(quiz_id);
CREATE INDEX IF NOT EXISTS idx_quiz_attempts_student_id ON quiz_attempts(student_id);
CREATE INDEX IF NOT EXISTS idx_quiz_attempts_quiz_student_status ON quiz_attempts(quiz_id, student_id, status);
CREATE INDEX IF NOT EXISTS idx_quiz_attempts_status_started_at ON quiz_attempts(status, started_at);
CREATE INDEX IF NOT EXISTS idx_student_responses_attempt_id ON student_responses(attempt_id);
CREATE INDEX IF NOT EXISTS idx_activity_logs_quiz_created_date ON activity_logs(quiz_id, created_date);
CREATE INDEX IF NOT EXISTS idx_activity_logs_attempt_created_date ON activity_logs(attempt_id, created_date);
