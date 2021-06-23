### 1. Recommendation system
Recommendation system được sử dụng trong nhiều nơi trong cuộc sống, có thể lấy ví dụ như hệ thống của YouTube, Shopee, Netflix.
Recommendation system gồm hai loại, Collaborative filtering và Content-based filtering.
Collaborative filtering là một ứng dụng thuật toán Matrix Factorization để  xác định mối liên hệ giữa user và item. Với đầu vào là các rating có sẵn của các user, hệ thống sẽ đưa ra dự đoán về rating cho các item khác mà user chưa đánh giá.

### 2. Matrix Factorization
Đầu vào của thuật toán là một utility matrix có kích thước n x m. Với n là số lượng n là số lượng user và m là số lượng item. Utility matrix này chứa tất cả các rating mà user đã đánh giá cho các items. Giả sử trong ma trận này chứa k latent feature, các feature này là cơ sở để chúng ta tính toán được các ratings cho các items khác mà không có sẵn. Thuật toán Matrix factorization sẽ tạo ra hai ma trận P (n x k) và Q (k x m)

![PxQ](/image/PxQ.png)

Mỗi hàng của P sẽ đại diện cho mối tương quan giữa users và items, tương tự mỗi hàng của Q đại diện cho mối tương quan giữa items và users. Để dự đoán được ratings của items [i] bởi user [j] ta có thể tính bằng cách nhân hai vector:

![rij](/image/rij.png)

Chúng ta có thể xác định hàm loss của thuật toán:

![eij](/image/loss.png)

Để tối ưu hóa hàm loss, chúng ta cần biết được xu hướng của hàm loss, từ đó thay đổi từng giá trị trong P và Q:

![update](/image/update.png)

Ta thực hiên thay đổi như sau:

![update2](/image/update1.png)



