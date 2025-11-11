package com.example.demo_for_yolo_int8;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.widget.TextView;

import com.example.demo_for_yolo_int8.databinding.ActivityMainBinding;

import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'demo_for_yolo_int8' library on application startup.

    private ActivityMainBinding binding;

    private Handler handler = null;

    private TextView tv;

    private final String OttOnnxName = "king_game_int8.onnx";

    private void run_ott_check(){
        // ott检测测试开始
        AssetManager assetManager = getAssets();
        
        // 使用try-with-resources自动关闭InputStream
        try (InputStream inputStream = assetManager.open("test_2400_1080.jpg")) {
            Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
            
            String onnxPath = FileHelper.writeAssetToInternalStorage(getApplication(), OttOnnxName);
            OttRunner ott = new OttRunner();
            ott.InitOtt(onnxPath);
            
            long startTime = System.currentTimeMillis();
            for(int i = 0; i<100; i++) {
                OttCheckAns[] ottans = ott.ProcessOtt(bitmap);
            }
            long endTime = System.currentTimeMillis();
            long duration = endTime - startTime;
    
            ott.DeInitOtt();
            handler.post(new Runnable() {
                @Override
                public void run() {
                    // 在主线程中更新 UI
                    tv.setText("100 time yolo test use: " + duration + " ms");
                }
            });
            
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    Thread thread1 = new Thread(new Runnable() {
        @Override
        public void run() {
            run_ott_check();
        }
    });

    private void check_ott(){
        thread1.start();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        handler = new Handler(Looper.getMainLooper());
        // Example of a call to a native method
        tv = binding.sampleText;
        tv.setText("init interface success start test");
        check_ott();
    }

}