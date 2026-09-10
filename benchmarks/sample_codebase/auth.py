def login_user(conn, username, password):
    """Authenticate a user against the database and create a session."""
    user = conn.fetch_one(
        "SELECT id, password_hash FROM users WHERE username = ?",
        (username,),
    )
    if user is None:
        return None

    if not verify_password(password, user["password_hash"]):
        return None

    session_id = conn.insert(
        "INSERT INTO sessions (user_id) VALUES (?)",
        (user["id"],),
    )
    return session_id


def logout_user(conn, token):
    """Terminate the current session in the database."""
    conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    return True


def reset_password(conn, email):
    """Send a password reset link and persist the token."""
    token = generate_reset_token(email)
    conn.execute(
        "UPDATE users SET reset_token = ? WHERE email = ?",
        (token, email),
    )
    conn.commit()
    return token


def change_password(conn, user_id, new_password):
    """Replace a user's password hash after validating the new password."""
    password_hash = hash_password(new_password)
    conn.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (password_hash, user_id),
    )
    conn.commit()
    return True


def get_user_profile(conn, user_id):
    """Load the profile and preferences associated with a user account."""
    return conn.fetch_one(
        "SELECT id, display_name, preferences FROM user_profiles WHERE user_id = ?",
        (user_id,),
    )
