### 1. Recommendation system
Recommendation system được sử dụng trong nhiều nơi trong cuộc sống, có thể lấy ví dụ như hệ thống của YouTube, Shopee, Netflix.
Recommendation system gồm hai loại, Collaborative filtering và Content-based filtering.
Collaborative filtering là một ứng dụng thuật toán Matrix Factorization để  xác định mối liên hệ giữa user và item. Với đầu vào là các rating có sẵn của các user, hệ thống sẽ đưa ra dự đoán về rating cho các item khác mà user chưa đánh giá.

### 2. Matrix Factorization
Đầu vào của thuật toán là một utility matrix có kích thước n x m. Với n là số lượng n là số lượng user và m là số lượng item. Utility matrix này chứa tất cả các rating mà user đã đánh giá cho các items. Giả sử trong ma trận này chứa k latent feature, các feature này là cơ sở để chúng ta tính toán được các ratings cho các items khác mà không có sẵn. Thuật toán Matrix factorization sẽ tạo ra hai ma trận P (n x k) và Q (k x m)

![PxQ](./image/PxQ.png)

Mỗi hàng của P sẽ đại diện cho mối tương quan giữa users và items, tương tự mỗi hàng của Q đại diện cho mối tương quan giữa items và users. Để dự đoán được ratings của items [i] bởi user [j] ta có thể tính bằng cách nhân hai vector:

![rij](./image/rij.png)

Chúng ta có thể xác định hàm loss của thuật toán:

![eij](./image/loss.png)

Để tối ưu hóa hàm loss, chúng ta cần biết được xu hướng của hàm loss, từ đó thay đổi từng giá trị trong P và Q:

![update](./image/update.png)

Ta thực hiên thay đổi như sau:

![update2](./image/update1.png)

![update3](./image/update2.png)

#### Regularization
Để tránh tình trạng overfiting, ta áp dụng thêm regularization vào công thức tính:

![regular](./image/regular.png)

Từ công thức trên, ta tính lại công thức update cho P và Q:

![regular-update](./image/regular1.png)

![regular-update](./image/regular2.png)

### 3.Usage
#### Read and convert dataframe to utility matrix
Read file csv
```
anime = pd.read_csv('/content/anime.csv')
rating = pd.read_csv('/content/rating.csv')
```

Convert

```
rating.rating.replace({-1: 0}, inplace = True)
data_sub = rating[rating['user_id']<=10000]
data_sub = np.array(data_sub)
rows, row_pos = np.unique(data_sub[:, 0], return_inverse=True)
cols, col_pos = np.unique(data_sub[:, 1], return_inverse=True)

pivot_table = np.zeros((len(rows), len(cols)), dtype=data_sub.dtype)
pivot_table[row_pos, col_pos] = data_sub[:, 2]
pivot_table.shape
```

Tính ma trận và vẽ đồ thị hàm loss

```
model = AnimeRecommendationSystem(utility_matrix=pivot_table)
model.matrix_factorization(iterations=200, lr=0.001, lambda_=0.02)
```



