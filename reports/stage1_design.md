# MIND-3 — المرحلة 1: التصميم وخطة التجربة

**التاريخ:** 2026-11-12 — بيئة معزولة: `.venv` (Python 3.12.13, numpy 2.5.4 فقط).

## 1. النطاق العلمي الصادق

البيانات صغيرة (عشرات الأمثلة) ⇒ لا يمكن تدريب نموذج لغة عام من الصفر. لذلك:
- **النموذج التجريبي MIND-3** = نموذج معرفة واستجابة قابل للتدريب يختبر الفرضيات الثلاث فعليًا.
- **لا يُستخدم أي نموذج مدرّب مسبقًا** في الخفاء. لا torch، لا transformers.
- القدرة اللغوية = أنماط أسئلة/أجوبة مدربّة (عربية/إنجليزية) + كشف لغة. المحادثة الحرة خارج النطاق ويُعلن ذلك.

## 2. كيف تدخل الفرضيات الثلاث في المعالجة فعليًا

### H1 — Reference-Based Token Representation
- `SymbolRegistry`: كل رمز (حرف) له معرف ثابت `<PAD>=0, <UNK>=1` ثم رموز الأبجدية. فصل صريح: Registry / Token ID / Embedding / Decoder.
- المسار الفعلي: نص → IDs → lookup في مصفوفة Embedding متعلّمة `E[V,d]` → متوسط = متجه السياق. كل الحسابات على IDs.
- Decoder يعيد IDs → نص (اختبار round-trip + تصادمات + عربية/إنجليزية/مسافات/ترقيم).
- قياس: حجم التمثيل بالذاكرة وسرعة الترميز مقابل vocab نصي تقليدي.

### H2 — Probabilistic Transition System (هجين: متعلّم + استدلال)
- جزء متعلّم: مصفوفة انتقالات `W[d,K]` + انحياز `b` فوق K وحدة معرفية (كل سجل = وحدة + وحدة "مجهول").
- `p(unit|context) = softmax(W·c + b)`.
- استدلال متعدد المسارات: beam بعرض k يحتفظ بعدة مسارات بأوزانها، تحقق Multi-PathVerification ضد مخزن المعرفة (ثقة السجل + توفر الإجابة باللغة + عتبة هامش)، وإحياء Revival للمسارات المستبعدة عند فشل التحقق. تسجيل سبب الاختيار النهائي.
- تمييز بحثي موثّق: `W` جزء من البنية المتعلمة؛ beam/verification/revival آلية استدلال مستقلة.

### H3 — Confidence-Weighted Markdown Training
- Pipeline: قراءة MD → تحليل Claim/Category/Confidence/Source/Verification → ربط المصدر → أمثلة تدريب (أنماط أسئلة AR/EN) → فقد Cross-Entropy موزون بالثقة `loss_i *= confidence_i` → تدريب فعلي (Adam, NumPy) → حفظ `model.npz + registry.json + units.json + training_log.json`.
- الثقة لا ترتفع بالتكرار؛ التدريب لا يعدّل السجلات. فصل ثقة صحة الادعاء عن موثوقية المصدر عن مستوى المهارة.

## 3. خط الأساس والـAblations (نفس البيانات)

نموذج واحد بأعلام إعدادات:
|Variant|Reference IDs|Transitions|Confidence weighting|
|---|---|---|---|
|Full MIND-3|✓|✓|✓|
|−Reference IDs|✗ (hashed BoW)|✓|✓|
|−Transitions|✓|✗ (greedy)|✓|
|−Confidence|✓|✓|✗ (weights=1)|
|Baseline|✗|✗|✗|

## 4. بنية الملفات

```
knowledge/records/*.md      سجلات المعرفة (مصدر التدريب)
knowledge/user_profile.md   نظرة عامة بصيغة التكليف
mind3/registry.py           H1
mind3/transitions.py        H2
mind3/knowledge.py          محلل MD
mind3/model.py              النموذج + الاستدلال (أعلام الـablation)
mind3/training.py           H3 pipeline
mind3/search.py             البحث عند الطلب
mind3/engine.py             التوجيه (معرفة/بحث) + التشخيص
server.py + static/         واجهة المحادثة المحلية
eval/run_eval.py            مجموعة التقييم + القياسات
artifacts/                  النموذج والسجل والإعدادات
reports/                    التقارير
```

## 5. خطة التنفيذ الصغيرة (بالترتيب)

1. Stage 2: سجلات MD (اسم، عمر متعارض، DOB، Python 50%، إنجليزية، جنس مجهول + الأساسيات الخمسة).
2. Stage 3: registry + اختبارات round-trip/تصادم/ذاكرة/سرعة.
3. Stage 4: transitions + اختبار "أنا اسمي…" / "مستواي في Python…".
4. Stage 5: تدريب فعلي وحفظ + منحنى فقد ودقة تدريب.
5. Stage 6: بحث حقيقي (Wikipedia + DDG) عند الطلب فقط.
6. Stage 7: واجهة محلية + اختبار متصفح حي.
7. Stage 8: تقييم آلي + ablations + قياس دقة/زمن/ذاكرة/إجابات غير مدعومة.
8. Stage 9: تحقق نهائي + تقرير يجيب عن الأسئلة العشرة بأدلة VERIFIED/UNVERIFIED.

## 6. معايير القياس

دقة الإجابات، نسبة الإجابات غير المدعومة (hallucination)، زمن الاستجابة (ms)، ذاكرة العملية (MB)، وأثر تعطيل كل مكون. النتائج الفعلية أساس الاستنتاج.