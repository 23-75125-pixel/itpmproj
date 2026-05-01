create table if not exists users (
    id varchar(50) primary key,
    email varchar(255) not null unique,
    full_name varchar(255) not null,
    role varchar(20) not null check (role in ('admin', 'user')),
    section_name varchar(255) not null default '',
    avatar_url text not null default '',
    password_hash text not null,
    created_at timestamp not null default now()
);

create table if not exists quizzes (
    id varchar(50) primary key,
    creator_id varchar(50) not null references users(id),
    title varchar(255) not null,
    description text,
    subject varchar(255) not null,
    time_limit_minutes int not null,
    status varchar(20) not null check (status in ('draft', 'published', 'closed')),
    quiz_code varchar(50) not null unique,
    monitoring_enabled boolean not null default false,
    assigned_section varchar(255) not null default '',
    scheduled_start timestamp null,
    scheduled_end timestamp null,
    created_at timestamp not null default now()
);

create table if not exists questions (
    id varchar(50) primary key,
    quiz_id varchar(50) not null references quizzes(id) on delete cascade,
    question_text text not null,
    question_type varchar(30) not null check (question_type in ('multiple_choice', 'true_false', 'short_answer')),
    points int not null default 1,
    sort_order int not null default 0,
    created_at timestamp not null default now()
);

create table if not exists question_options (
    id varchar(50) primary key,
    question_id varchar(50) not null references questions(id) on delete cascade,
    option_text text not null,
    is_correct boolean not null default false,
    sort_order int not null default 0,
    created_at timestamp not null default now()
);

create table if not exists quiz_attempts (
    id varchar(50) primary key,
    quiz_id varchar(50) not null references quizzes(id) on delete cascade,
    student_id varchar(50) not null references users(id),
    quiz_code varchar(50) not null,
    score int not null default 0,
    percentage numeric(5, 2) not null default 0,
    status varchar(20) not null check (status in ('in_progress', 'submitted', 'auto_submitted')),
    started_at timestamp not null,
    submitted_at timestamp null,
    consent_given boolean not null default false,
    created_at timestamp not null default now()
);

create table if not exists student_responses (
    id varchar(50) primary key,
    attempt_id varchar(50) not null references quiz_attempts(id) on delete cascade,
    question_id varchar(50) not null references questions(id) on delete cascade,
    selected_option varchar(255),
    text_response text,
    is_correct boolean not null default false,
    created_at timestamp not null default now()
);

create table if not exists activity_logs (
    id varchar(50) primary key,
    quiz_id varchar(50) not null references quizzes(id) on delete cascade,
    attempt_id varchar(50) not null references quiz_attempts(id) on delete cascade,
    event_type varchar(50) not null,
    event_description text not null,
    flag_level varchar(20) not null default 'low' check (flag_level in ('low', 'medium', 'high')),
    reviewed boolean not null default false,
    instructor_notes text,
    created_date timestamp not null default now()
);

create index if not exists idx_quizzes_creator_id on quizzes(creator_id);
create index if not exists idx_questions_quiz_id on questions(quiz_id);
create index if not exists idx_question_options_question_id on question_options(question_id);
create index if not exists idx_quiz_attempts_quiz_student on quiz_attempts(quiz_id, student_id);
create index if not exists idx_student_responses_attempt_id on student_responses(attempt_id);
create index if not exists idx_activity_logs_quiz_id on activity_logs(quiz_id);
create index if not exists idx_activity_logs_attempt_id on activity_logs(attempt_id);
