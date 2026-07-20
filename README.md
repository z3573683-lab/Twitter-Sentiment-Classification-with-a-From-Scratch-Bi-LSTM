# Sentiment Analysis with a From-Scratch Bidirectional LSTM
# تحليل المشاعر باستخدام Bidirectional LSTM مبني من الصفر

A binary sentiment classifier trained end-to-end on the **Sentiment140** dataset (1.6M tweets), using a **trainable embedding layer built from scratch** (no pretrained word vectors) feeding a **Bidirectional LSTM** network.

موديل تصنيف مشاعر ثنائي (إيجابي/سلبي) اتدرب بالكامل على داتاست **Sentiment140** (1.6 مليون تغريدة)، باستخدام طبقة **Embedding قابلة للتدريب من الصفر** (من غير أي Vectors جاهزة زي GloVe)، متبوعة بشبكة **Bidirectional LSTM**.

---

## 1. Project Overview | نظرة عامة على المشروع

| Item / البند | Detail / التفاصيل |
|---|---|
| Task / المهمة | Binary sentiment classification (Positive/Negative) — تصنيف ثنائي (إيجابي/سلبي) |
| Dataset / الداتاست | Sentiment140 — 1,600,000 tweets / 1.6 مليون تغريدة |
| Framework / الأداة | TensorFlow / Keras |
| Core architecture / المعمارية الأساسية | `Embedding` → `Bidirectional LSTM` → `GlobalMaxPooling1D` → `Dense` |
| Approach / المنهج | Trained from scratch — embeddings learned during training, not loaded from GloVe/Word2Vec / متدرب من الصفر بدون أي Embeddings جاهزة |

---

## 2. Dataset | البيانات

**EN:** Source file: `training.1600000.processed.noemoticon.csv`. The dataset contains 1,600,000 tweets, perfectly balanced — 800,000 negative (`0`) and 800,000 positive (`4`, remapped to `1`). Columns used: `text` (raw tweet) and `target` (label).

**AR:** ملف المصدر: `training.1600000.processed.noemoticon.csv`. الداتاست فيه 1,600,000 تغريدة متوازنة تمامًا — 800,000 سلبي (`0`) و800,000 إيجابي (`4`، اتحول لـ `1`). الأعمدة المستخدمة: `text` (نص التغريدة) و`target` (التصنيف).

### Train / Validation / Test Split | تقسيم البيانات

**EN:** A single **stratified** split was performed once (`random_state=42`) and used consistently throughout the project — no re-splitting at any later stage.

**AR:** تم عمل تقسيم **طبقي (Stratified)** واحد بس (`random_state=42`) واتستخدم بثبات في كل المشروع — من غير أي إعادة تقسيم لاحقًا.

| Split / القسم | Size / الحجم | Percentage / النسبة |
|---|---|---|
| Train / تدريب | 1,280,000 | 80% |
| Validation / تحقق | 160,000 | 10% |
| Test / اختبار | 160,000 | 10% |

---

## 3. Preprocessing Pipeline | خط معالجة النصوص

**EN:** Implemented in `advanced_normalization()`:
1. Lowercasing all text
2. Removing URLs (`http(s)://...`, `www...`)
3. Removing `@mentions`
4. Stripping the `#` symbol while keeping the hashtag word itself
5. Collapsing repeated characters — e.g. `looooove` → `loove` (keeps emphasis signal without exploding vocabulary)
6. Removing all non-alphabetic characters
7. Removing English stopwords **except negation words** (`not`, `no`, `never`, `neither`, `nor`, `but`) — preserving polarity-critical terms
8. Normalizing whitespace

**AR:** مطبقة جوه دالة `advanced_normalization()`:
1. توحيد كل الحروف لحالة صغيرة (Lowercasing)
2. إزالة الروابط (`http(s)://...`, `www...`)
3. إزالة الـ `@mentions`
4. إزالة علامة `#` فقط مع الاحتفاظ بكلمة الهاشتاج نفسها
5. تقليل الحروف المكررة — مثلاً `looooove` → `loove` (بيحافظ على إشارة التأكيد من غير ما يضخم القاموس)
6. إزالة أي رموز غير حروف إنجليزية
7. إزالة كلمات الوقف الإنجليزية **ما عدا كلمات النفي** (`not`, `no`, `never`, `neither`, `nor`, `but`) — عشان معنى الجملة (إيجابي/سلبي) ميضيعش
8. توحيد المسافات

### Tokenization & Padding | التحويل لرموز والحشو

**EN/AR:** `Tokenizer(num_words=50000, oov_token="<OOV>")` — الجمل بتتحول لأرقام وبعدين بتتحشى/تتقص لطول ثابت **100 توكن** (`padding='pre'`, `truncating='post'`).

---

## 4. Model Architecture | معمارية الموديل

```
Embedding(vocab_size=50000, embedding_dim=128)
        ↓
Bidirectional(LSTM(64, return_sequences=True))
        ↓
GlobalMaxPooling1D()
        ↓
Dense(32, activation='relu')
        ↓
Dropout(0.5)
        ↓
Dense(1, activation='sigmoid')
```

**EN:** Loss: Binary Crossentropy | Optimizer: Adam | Metric: Accuracy

**AR:** دالة الخسارة: Binary Crossentropy | المُحسِّن (Optimizer): Adam | مقياس التقييم: Accuracy

### Training Configuration | إعدادات التدريب

**EN:**
- Epochs: up to 10 (with Early Stopping)
- Batch size: 1024
- Callbacks:
  - `EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)`
  - `ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-5)`
  - `ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)`

**AR:**
- عدد الدورات (Epochs): حتى 10 (مع EarlyStopping)
- حجم الدفعة (Batch size): 1024
- الأدوات المساعدة (Callbacks):
  - `EarlyStopping` بيوقف التدريب لو الـ val_loss متحسنش لمدة 3 دورات، ويرجع أفضل أوزان
  - `ReduceLROnPlateau` بيقلل معدل التعلم لو الموديل "علّق"
  - `ModelCheckpoint` بيحفظ أفضل نسخة من الموديل بس

---

## 5. Training Results | نتائج التدريب

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss | Learning Rate |
|---|---|---|---|---|---|
| 1 | 0.7887 | 0.4566 | 0.8090 | 0.4157 | 0.0010 |
| 2 | 0.8197 | 0.4023 | **0.8126** | **0.4087** ✅ best | 0.0010 |
| 3 | 0.8338 | 0.3742 | 0.8122 | 0.4186 | 0.0010 |
| 4 | 0.8484 | 0.3441 | 0.8080 | 0.4380 | 0.0010 |
| 5 | 0.8733 | 0.2932 | 0.8039 | 0.4886 | 0.0002 (reduced) |

**EN — Reading the curves:**
- Validation loss reached its minimum at **epoch 2** (`0.4087`), then increased for 3 consecutive epochs → `EarlyStopping` (patience=3) halted training after epoch 5.
- Training accuracy kept climbing (up to 87.3% by epoch 5) while validation accuracy declined after epoch 2 — a clear **overfitting signature**.
- Because `restore_best_weights=True`, the **final saved model corresponds to epoch 2's weights**, not epoch 5's — this is why the test results below closely match epoch 2's validation numbers.

**AR — تفسير المنحنيات:**
- أقل قيمة لـ val_loss كانت عند **الـ Epoch التاني** (`0.4087`)، وبعدها الرقم زاد لـ 3 دورات متتالية → `EarlyStopping` (بصبر 3 دورات) أوقف التدريب بعد Epoch 5.
- دقة التدريب فضلت تزيد (لحد 87.3% في Epoch 5) بينما دقة التحقق قلّت بعد Epoch 2 — ده **مؤشر واضح على Overfitting** (الموديل بدأ يحفظ بيانات التدريب بدل ما يتعمم).
- بما إن `restore_best_weights=True`، **الموديل النهائي المحفوظ هو أوزان Epoch 2** مش Epoch 5 — وده سبب تطابق نتائج الاختبار تحت مع أرقام Epoch 2 مش الأخير.

### Accuracy Curve | منحنى الدقة
![Training vs Validation Accuracy](assets/accuracy_curve.png)

**EN:** The gap that opens up between the training (blue) and validation (green) curves after epoch 2 is the visual signature of overfitting described above.
**AR:** الفجوة اللي بتتفتح بين منحنى التدريب (الأزرق) ومنحنى التحقق (الأخضر) بعد Epoch 2 هي الدليل البصري على الـ Overfitting اللي اتكلمنا عنه فوق.

### Loss Curve | منحنى الخسارة
![Training vs Validation Loss](assets/loss_curve.png)

**EN:** Training loss (red) keeps falling throughout, while validation loss (orange) bottoms out at epoch 2 then rises — the clearest single indicator of where to stop training.
**AR:** خسارة التدريب (الأحمر) فضلت تقل طول الوقت، بينما خسارة التحقق (البرتقالي) وصلت لأقل نقطة عند Epoch 2 وبعدين بدأت تزيد — وده أوضح مؤشر على النقطة المفروض التدريب يوقف عندها.

---

## 6. Test Set Evaluation | تقييم بيانات الاختبار

**EN/AR:** Evaluated once on the untouched 160,000-tweet test split — اتقيّم مرة واحدة على الـ 160,000 تغريدة اختبار اللي الموديل ماشافهاش خالص:

```
              precision    recall  f1-score   support

Negative (0)       0.80      0.83      0.82     80000
Positive (1)       0.83      0.79      0.81     80000

    accuracy                           0.81    160000
   macro avg       0.81      0.81      0.81    160000
weighted avg       0.81      0.81      0.81    160000
```

**EN — Interpretation:**
- Overall **test accuracy: 81%**, consistent with the restored epoch-2 weights (val_accuracy 81.26%) — confirming Early Stopping worked as intended and the reported number isn't inflated by overfit epochs.
- The model is **slightly more recall-oriented on Negative** (0.83) and **slightly more precision-oriented on Positive** (0.83) — a small, not concerning, asymmetry.
- Errors are spread fairly evenly between false positives and false negatives (see confusion matrix in the notebook), with no class collapse.

**AR — التفسير:**
- **دقة الاختبار الإجمالية: 81%**، متطابقة تقريبًا مع val_accuracy بتاعة Epoch 2 (81.26%) — ده بيأكد إن EarlyStopping اشتغل صح والرقم مش منتفخ من دورات فيها Overfitting.
- الموديل **أميل شوية للـ Recall في الفئة السلبية** (0.83) **وللـ Precision في الفئة الإيجابية** (0.83) — فرق بسيط ومش مقلق.
- الأخطاء موزعة بشكل متوازن تقريبًا بين False Positive وFalse Negative، من غير انحياز واضح لفئة واحدة.

### Confusion Matrix | مصفوفة الارتباك
![Confusion Matrix](assets/confusion_matrix.png)

**EN:** Visual confirmation of the classification report above — errors (off-diagonal cells) are close in magnitude on both sides, meaning the model doesn't systematically favor one class over the other.
**AR:** تأكيد بصري لتقرير التصنيف فوق — الأخطاء (الخلايا خارج القطر) متقاربة في الحجم من الجهتين، يعني الموديل مش منحاز بشكل ممنهج لفئة على حساب التانية.

---

## 7. Qualitative Inference Examples | أمثلة استدلال حقيقية

| Input / المدخل | Predicted / التوقع | Confidence / الثقة |
|---|---|---|
| "I really love this new update, it is amazing!" | Positive 😊 | 98.70% |
| "I hate waiting for so long, this is the worst service ever." | Negative 😠 | 98.26% |
| "The weather is okay, but the food was bad." | Negative 😠 | 88.09% |

**EN:** The third example is the most interesting — a **mixed-sentiment sentence** ("okay" + "bad") correctly resolved toward the dominant negative clause, showing the negation/contrast-preserving stopword strategy (keeping "but") is paying off.

**AR:** المثال التالت هو الأهم — جملة **مشاعر مختلطة** ("okay" و"bad" مع بعض) واتحلّت صح لصالح الجزء السلبي الأقوى، وده بيوضح إن استراتيجية الاحتفاظ بكلمات النفي/التضاد زي "but" فعلاً بتفرق.

---

## 8. Known Limitations | نقاط الضعف الحالية

**EN:**
- **No baseline comparison** — no classical ML model (e.g. TF-IDF + Logistic Regression) or alternative architecture (GRU, Transformer) was benchmarked, so 81% accuracy can't yet be judged in relative terms.
- **No hyperparameter search** — embedding dimension (128), LSTM units (64), batch size (1024) were fixed, not tuned.
- **No pretrained embeddings comparison** — a GloVe/Word2Vec-initialized run would clarify how much the from-scratch embedding layer is actually contributing.
- **No error analysis** — misclassified examples weren't inspected for systematic failure patterns (sarcasm, negation scope, short tweets).
- **Domain-specific data** — trained purely on 2009-era Twitter text; generalization to other domains is untested.

**AR:**
- **مفيش مقارنة Baseline** — مفيش موديل كلاسيكي (TF-IDF + Logistic Regression) ولا معمارية بديلة (GRU, Transformer) اتقارنت بيه، فمعرفش أحكم على الـ 81% كويسة قد إيه نسبيًا.
- **مفيش Hyperparameter Search** — أبعاد الـ Embedding (128)، عدد وحدات الـ LSTM (64)، حجم الدفعة (1024) كلهم قيم ثابتة من غير تجربة.
- **مفيش مقارنة مع Embeddings جاهزة** — تجربة بـ GloVe/Word2Vec هتوضح فعلاً قد إيه طبقة الـ Embedding المتدربة من الصفر بتضيف قيمة.
- **مفيش Error Analysis** — الأمثلة اللي اتصنفت غلط ماتفحصتش عشان نلاقي نمط واضح للأخطاء (سخرية، نطاق النفي، تغريدات قصيرة).
- **بيانات محددة المجال** — الموديل اتدرب بس على تغريدات من 2009، والتعميم على مجالات تانية (مراجعات، أخبار، نصوص رسمية) لسه ماتاختبرش.

## 9. Suggested Next Steps | خطوات مقترحة للتطوير

**EN:**
1. Add a classical ML baseline (TF-IDF + Logistic Regression/SVM) for a fair comparison point.
2. Run a small hyperparameter sweep (embedding dim, LSTM units, dropout rate).
3. Compare against GloVe-initialized embeddings.
4. Add qualitative error analysis on a sample of misclassified test tweets.

**AR:**
1. ضيف Baseline كلاسيكي (TF-IDF + Logistic Regression/SVM) عشان نقطة مقارنة عادلة.
2. جرب Hyperparameter Search بسيط (أبعاد Embedding، عدد وحدات LSTM، نسبة Dropout).
3. قارن مع Embeddings جاهزة زي GloVe.
4. اعمل تحليل نوعي للأخطاء على عينة من التغريدات اللي اتصنفت غلط.

---

## 10. How to Reproduce | إزاي تشغّل المشروع

```bash
pip install tensorflow nltk pandas scikit-learn matplotlib seaborn
```

**EN:** Run the notebook top to bottom in a GPU-enabled environment (Google Colab recommended). The trained model is saved as `sentiment_model.keras`.

**AR:** شغّل النوت بوك من الأول للآخر في بيئة فيها GPU (يُفضّل Google Colab). الموديل بعد التدريب بيتحفظ باسم `sentiment_model.keras`.
