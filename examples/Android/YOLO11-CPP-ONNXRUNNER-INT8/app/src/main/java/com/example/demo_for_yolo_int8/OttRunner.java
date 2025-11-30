package com.example.demo_for_yolo_int8;

import android.graphics.Bitmap;

import java.util.concurrent.locks.ReentrantLock;

public class OttRunner {
    static {
       System.loadLibrary("demo_for_onnx_int8");
    }
    private boolean is_init = false;
    private final ReentrantLock lock = new ReentrantLock();

    private long obj = 0;

    public boolean InitOtt(String onnxpath){
        synchronized(lock){
            if (is_init) {
                return true;
            }
            obj = Init(onnxpath); // 调用此方法前需要先释放onnx文件
            if(obj == 0){
                is_init = false;
                return is_init;
            }
            is_init = true;
            return is_init;
        }
    }

    public boolean DeInitOtt(){
        synchronized (lock){
            if(!is_init){
                return true;
            }
            Dinit(obj);
            obj = 0;
            is_init = false;
            return true;
        }
    }

    public OttCheckAns[] ProcessOtt(Bitmap bitmap){
        synchronized (lock){
            if(!is_init){
                return null;
            }
            OttCheckAns[] array = Process(obj, bitmap);
            return array;
        }
    }

    private native long Init(String onnxpath);

    private native void Dinit(long obj);

    private native OttCheckAns[] Process(long obj, Bitmap bitmap);
}
