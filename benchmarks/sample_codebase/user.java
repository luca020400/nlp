public class UserService {
    /** Update user profile details in the database. */
    public boolean updateProfile(UserRepository repo, String userId) {
        UserProfile profile = repo.findById(userId);
        if (profile == null) {
            return false;
        }
        profile.setUpdatedAt(System.currentTimeMillis());
        repo.save(profile);
        return true;
    }

    /** Delete a user account and its related rows. */
    public boolean deleteAccount(UserRepository repo, String userId) {
        repo.deleteSessions(userId);
        repo.deleteById(userId);
        return true;
    }

    /** Find an account by its email address for account recovery. */
    public User findByEmail(UserRepository repo, String email) {
        return repo.findByEmail(email);
    }

    /** Create a profile record with default notification preferences. */
    public boolean createProfile(UserRepository repo, String userId) {
        UserProfile profile = new UserProfile(userId);
        profile.setNotificationsEnabled(true);
        repo.save(profile);
        return true;
    }

    /** List active sessions so they can be revoked from account settings. */
    public List<Session> listActiveSessions(UserRepository repo, String userId) {
        return repo.findActiveSessions(userId);
    }
}
