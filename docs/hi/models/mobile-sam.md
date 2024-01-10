---
comments: true
description: Ultralytics मार्गदर्शिका में MobileSAM के बारे में और उसके प्रायोगशाला तुलनात्मक विवेचन, मूल SAM के साथ तुलना और इसे Ultralytics ढांचे में डाउनलोड और परीक्षण कैसे करें। अपने मोबाइल ऐप्लिकेशन को बेहतर बनाएं।
keywords: MobileSAM, Ultralytics, SAM, मोबाइल ऐप्लिकेशन, Arxiv, GPU, API, छवि एनकोडर, मास्क डिकोडर, मॉडल डाउनलोड, परीक्षण पद्धति
---

![MobileSAM लोगो](https://github.com/ChaoningZhang/MobileSAM/blob/master/assets/logo2.png?raw=true)

# मोबाइल सेगमेंट कुछ भी (MobileSAM)

मोबाइलSAM पेपर [arXiv](https://arxiv.org/pdf/2306.14289.pdf) पर अब उपलब्ध है।

MobileSAM के संचालन का एक प्रदर्शन कम्प्यूटर पर पहुंचा जा सकता है उस [डेमो लिंक](https://huggingface.co/spaces/dhkim2810/MobileSAM) के माध्यम से। Mac i5 CPU पर प्रदर्शन करने में लगभग 3 सेकंड का समय लगता है। हगिंग फेस डेमो परिवेश और कम प्रदर्शन वाले सीपियू ने प्रतिक्रिया को धीमी किया है, लेकिन यह अभी भी प्रभावी ढंग से काम करता है।

मोबाइलSAM [Grounding-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), [AnyLabeling](https://github.com/vietanhdev/anylabeling), और [Segment Anything in 3D](https://github.com/Jumpat/SegmentAnythingin3D) सहित विभिन्न परियोजनाओं में लागू है।

मोबाइलSAM एक एकल GPU पर 100k डेटासेट (मूल छवि का 1%) के साथ प्रशिक्षित होता है और इसमें एक दिन से कम समय लगता है। इस प्रशिक्षण के लिए कोड भविष्य में उपलब्ध कराया जाएगा।

## उपलब्ध मॉडल, समर्थित कार्य और ऑपरेटिंग मोड

इस तालिका में उपलब्ध मॉडल, उनके विशिष्ट पूर्व-प्रशिक्षित वजन, वे कार्य जिन्हें वे समर्थन करते हैं, और उनका अभिन्नतम संगतता के साथ विभिन्न ऑपरेटिंग मोड (इंफरेंस, वैधानिकी, प्रशिक्षण, और निर्यात) प्रदर्शित किए गए हैं, जिन्हें समर्थित मोड के लिए ✅ emoji और असमर्थित मोड के लिए ❌ emoji से दर्शाया गया है।

| मॉडल प्रकार | पूर्व-प्रशिक्षित वजन | समर्थित कार्य                              | इंफरेंस | वैधानिकी | प्रशिक्षण | निर्यात |
|-------------|----------------------|--------------------------------------------|---------|----------|-----------|---------|
| MobileSAM   | `mobile_sam.pt`      | [इंस्टेंस सेगमेंटेशन](../tasks/segment.md) | ✅       | ❌        | ❌         | ✅       |

## SAM से MobileSAM में अनुकूलन

MobileSAM मूल SAM की तरफ से समान पाइपलाइन बरकरार रखता है, हमने मूल की प्री-प्रोसेसिंग, पोस्ट-प्रोसेसिंग और सभी अन्य इंटरफेसों को सम्मिलित कर दिया है। इसलिए, वर्तमान में मूल SAM का उपयोग करने वाले लोग मिनिमल प्रयास के साथ MobileSAM में ट्रांसिशन कर सकते हैं।

MobileSAM मूल SAM के समान पाइपलाइन में उत्तम प्रदर्शन करता है और केवल छवि एन्कोडर में परिवर्तन होता है। विशेष रूप से, हम मूल भारीवज्ञानिक ViT-H एन्कोडर (632M) को एक छोटे Tiny-ViT (5M) से बदलते हैं। एकल GPU पर MobileSAM लगभग 12ms प्रति छवि पर ऑपरेट करता है: 8ms छवि एन्कोडर पर और 4ms मास्क डिकोडर पर।

विट-आधारित इमेज एन्कोडरों की तुलना नीचे दी गई तालिका प्रदान करती है:

| छवि एन्कोडर | मूल SAM | MobileSAM |
|-------------|---------|-----------|
| पैरामीटर्स  | 611M    | 5M        |
| स्पीड       | 452ms   | 8ms       |

मूल SAM और MobileSAM दोनों में समान प्रॉम्प्ट गाइडेड मास्क डिकोडर का उपयोग किया गया है:

| मास्क डिकोडर | मूल SAM | MobileSAM |
|--------------|---------|-----------|
| पैरामीटर्स   | 3.876M  | 3.876M    |
| स्पीड        | 4ms     | 4ms       |

यहां पाइपलाइन की तुलना है:

| पूरा पाइपलाइन (एन्कोडर+डिकोडर) | मूल SAM | MobileSAM |
|--------------------------------|---------|-----------|
| पैरामीटर्स                     | 615M    | 9.66M     |
| स्पीड                          | 456ms   | 12ms      |

MobileSAM और मूल SAM के प्रदर्शन को एक बिन्दु और बॉक्स के रूप में प्रदर्शित किया जाता है।

![बिन्दु के रूप में छवि](https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/assets/mask_box.jpg?raw=true)

![बॉक्स के रूप में छवि](https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/assets/mask_box.jpg?raw=true)

बेहतर प्रदर्शन से MobileSAM मौजूदा FastSAM की तुलना में लगभग 5 गुना छोटा और 7 गुना तेज है। अधिक विवरण [MobileSAM प्रोजेक्ट पेज](https://github.com/ChaoningZhang/MobileSAM) पर उपलब्ध हैं।

## Ultralytics में MobileSAM का परीक्षण

मूल SAM की तरह ही, हम Ultralytics में एक सीधा परीक्षण विधि प्रदान करते हैं, जिसमें बिंदु और बॉक्स प्रॉम्प्ट्स दोनों के लिए मोड शामिल हैं।

### मॉडल डाउनलोड

आप यहां से मॉडल डाउनलोड कर सकते हैं [here](https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt)।

### बिंदु प्रॉम्प्ट

!!! Example "उदाहरण"

    === "Python"
        ```python
        from ultralytics import SAM

        # मॉडल लोड करें
        model = SAM('mobile_sam.pt')

        # बिंदु प्रॉम्प्ट पर आधारित एक सेगमेंट पूर्वानुमान करें
        model.predict('ultralytics/assets/zidane.jpg', points=[900, 370], labels=[1])
        ```

### बॉक्स प्रॉम्प्ट

!!! Example "उदाहरण"

    === "Python"
        ```python
        from ultralytics import SAM

        # मॉडल लोड करें
        model = SAM('mobile_sam.pt')

        # बॉक्स प्रॉम्प्ट पर आधारित एक सेगमेंट पूर्वानुमान करें
        model.predict('ultralytics/assets/zidane.jpg', bboxes=[439, 437, 524, 709])
        ```

हमने `MobileSAM` और `SAM` दोनों को एक ही API का उपयोग करके इम्प्लिमेंट किया है। अधिक उपयोग जानकारी के लिए, कृपया [SAM पेज](sam.md) देखें।

## संदर्भ और आभार

अगर आप अपने अनुसंधान या विकास कार्य में MobileSAM का उपयोगयोगी पाते हैं, तो कृपया हमारे पेपर को साइट करने का विचार करें:

!!! Quote ""
=== "BibTeX"

        ```bibtex
        @article{mobile_sam,
          title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
          author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung Ho and Lee, Seungkyu and Hong, Choong Seon},
          journal={arXiv preprint arXiv:2306.14289},
          year={2023}
        }
