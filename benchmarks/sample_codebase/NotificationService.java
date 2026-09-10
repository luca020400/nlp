public class NotificationService {
    /** Queue an email with the receipt for a completed purchase. */
    public String sendReceipt(MailQueue queue, Order order) {
        String subject = "Receipt for order " + order.getId();
        String body = "Total paid: " + order.getTotal();
        return queue.enqueue(order.getEmail(), subject, body);
    }

    /** Disable promotional email for a subscriber while retaining the account. */
    public boolean unsubscribe(SubscriberRepository repo, String email) {
        Subscriber subscriber = repo.findByEmail(email);
        if (subscriber == null) {
            return false;
        }
        subscriber.setMarketingEnabled(false);
        repo.save(subscriber);
        return true;
    }

    /** Retry failed deliveries that have not reached the attempt limit. */
    public int retryFailedMessages(MailQueue queue, int maxAttempts) {
        int count = 0;
        for (Message message : queue.findFailed()) {
            if (message.getAttempts() < maxAttempts) {
                queue.requeue(message);
                count++;
            }
        }
        return count;
    }
}
