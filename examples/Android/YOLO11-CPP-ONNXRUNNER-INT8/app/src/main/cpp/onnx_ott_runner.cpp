#include <jni.h>
#include <android/bitmap.h>
#include <opencv2/opencv.hpp>
#include "android/log.h"
#include <sstream>
#include <string>
#include <iostream>
#include <fstream>
#include "ott_check_for_int8.h"
#define LOG_TAG "demo_for_onnx_int8"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

namespace {
    int GetBigCoreCountByCapacity() {
        int big_cores = 0;
        char path[128];
        FILE* fp;

        for (int i = 0; i < 32; i++) {
            // read cpu capacity
            snprintf(path, sizeof(path),
                     "/sys/devices/system/cpu/cpu%d/cpu_capacity", i);

            fp = fopen(path, "r");
            if (!fp) continue;

            int capacity = 0;
            fscanf(fp, "%d", &capacity);
            fclose(fp);

            // big core is big than 500
            if (capacity > 500) {
                big_cores++;
            }
        }

        return big_cores;
    }

    cv::Mat ReadIntArraryToMat(JNIEnv *env, jobject bitmap){
        cv::Mat dstMat;
        // read Bitmap from java
        AndroidBitmapInfo info;
        void *pixels;
        uint8_t *srcData;
        if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) {
            return dstMat;
        }

        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
            return dstMat;
        }

        if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
            return dstMat;
        }

        // cov Bitmap to OpenCV Mat
        cv::Mat srcMat(info.height, info.width, CV_8UC4, pixels);
        // cov mat as opencv read image
        cv::cvtColor(srcMat, dstMat, cv::COLOR_RGBA2BGR);

        // unlock Bitmap
        AndroidBitmap_unlockPixels(env, bitmap);
        return dstMat;
    }


    cv::Mat ReadBitmapWithAlpha(JNIEnv *env, jobject bitmap) {
        AndroidBitmapInfo info;
        void *pixels;
        if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) {
            return cv::Mat();
        }

        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
            return cv::Mat();
        }

        if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
            return cv::Mat();
        }

        cv::Mat rgba(info.height, info.width, CV_8UC4, pixels);
        cv::Mat result;
        rgba.copyTo(result);

        AndroidBitmap_unlockPixels(env, bitmap);
        return result;
    }

    jboolean ToJBool(bool value) {
        return value ? JNI_TRUE : JNI_FALSE;
    }

    bool ToCppBool(jboolean value) {
        return value == JNI_TRUE;
    }

    jstring ToJString(JNIEnv* env, const std::string& value) {
        return env->NewStringUTF(value.c_str());
    }

    std::string ToCppString(JNIEnv* env, jstring value) {
        jboolean isCopy;
        const char* c_value = env->GetStringUTFChars(value, &isCopy);
        if (c_value == nullptr) {
            return std::string();
        }
        std::string result(c_value);
        env->ReleaseStringUTFChars(value, c_value);
        return result;
    }

    std::vector<char> GetJavaByte(JNIEnv *env, jbyteArray array){
        jsize length = env->GetArrayLength(array);
        jbyte* bytes = env->GetByteArrayElements(array, nullptr);
        std::vector<char> retans(bytes, bytes + length);
        env->ReleaseByteArrayElements(array, bytes, 0);
        return retans;
    }

    jobjectArray ToJavaOttCheckAns(JNIEnv *env, const std::vector<OttCheckAns>& input){
        jclass ottClass = env->FindClass("com/example/demo_for_yolo_int8/OttCheckAns");
        if (!ottClass) {
            return nullptr;
        }
        jmethodID ottCheckAnsCtor = env->GetMethodID(ottClass, "<init>",
                                           "(DDDDLjava/lang/String;D)V");
        if(!ottCheckAnsCtor){
            return nullptr;
        }
        jobjectArray resultArray = env->NewObjectArray(input.size(), ottClass, nullptr);
        if (!resultArray) {
            return nullptr;
        }
        for (size_t i = 0; i < input.size(); ++i) {
            const OttCheckAns& item = input[i];

            jstring jBoxName = env->NewStringUTF(item.boxName.c_str());
            if (!jBoxName) continue;

            jobject ottObj = env->NewObject(ottClass, ottCheckAnsCtor,
                                            double(item.startPoint.x),
                                            double(item.startPoint.y),
                                            double(item.endPoint.x),
                                            double(item.endPoint.y),
                                            jBoxName,
                                            item.score);

            if (!ottObj) {
                env->DeleteLocalRef(jBoxName);
                continue;
            }

            env->SetObjectArrayElement(resultArray, i, ottObj);

            env->DeleteLocalRef(ottObj);
            env->DeleteLocalRef(jBoxName);
        }
        return resultArray;
    }

}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_example_demo_1for_1yolo_1int8_OttRunner_Init(JNIEnv *env, jobject thiz, jstring onnxpath) {
    // TODO: implement Init()
    if (onnxpath == nullptr) {
        LOGE("onnxpath object is null");
        return reinterpret_cast<jlong>(nullptr);
    }
    int bit_core_number = GetBigCoreCountByCapacity();
    std::string onnxpath_str = ToCppString(env, onnxpath);
    std::ifstream stream(onnxpath_str, std::ios::in | std::ios::binary);
    std::vector<char> model_bytes((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    auto* obj = new OttCheckForInt8();
    bool is_init_success = obj->Init(model_bytes, bit_core_number);
    if (is_init_success){
        return reinterpret_cast<jlong>(obj);
    }
    return reinterpret_cast<jlong>(nullptr);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_demo_1for_1yolo_1int8_OttRunner_Dinit(JNIEnv *env, jobject thiz, jlong obj) {
    auto* ptr = reinterpret_cast<OttCheckForInt8*>(obj);
    if(!ptr){
        return;
    }
    delete ptr;
}
extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_example_demo_1for_1yolo_1int8_OttRunner_Process(JNIEnv *env, jobject thiz, jlong obj,
                                                         jobject bitmap) {
    auto* ptr = reinterpret_cast<OttCheckForInt8*>(obj);
    if(!ptr){
        return nullptr;
    }
    if (bitmap == nullptr){
        return nullptr;
    }
    auto input_img= ReadIntArraryToMat(env, bitmap);
    std::vector<OttCheckAns> cppretans;
    if(!ptr->Process(input_img,cppretans)){
        return nullptr;
    }
    jobjectArray ret_ans = ToJavaOttCheckAns(env, cppretans);
    return ret_ans;
}