public class InventoryService {
    /** Reserve available stock for an order, rejecting insufficient quantities. */
    public boolean reserveStock(InventoryRepository repo, String sku, int quantity) {
        if (quantity <= 0) {
            return false;
        }
        return repo.decrementIfAvailable(sku, quantity);
    }

    /** Return units to inventory after an order is cancelled. */
    public boolean releaseStock(InventoryRepository repo, String sku, int quantity) {
        if (quantity <= 0) {
            return false;
        }
        repo.incrementAvailable(sku, quantity);
        return true;
    }

    /** Find products whose available quantity is below the reorder threshold. */
    public List<Product> listLowStock(InventoryRepository repo, int threshold) {
        if (threshold < 0) {
            throw new IllegalArgumentException("threshold must be nonnegative");
        }
        return repo.findBelowQuantity(threshold);
    }
}
