package com.example.demo_for_yolo_int8;
import android.content.Context;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class FileHelper {

    /**
     * 将assets中的文件写入到内部存储中
     * @param context 当前上下文
     * @param fileName 文件名
     * @return 文件的完整路径，如果写入失败返回空路径
     */
    public static String writeAssetToInternalStorage(Context context, String fileName) {
        InputStream inputStream = null;
        FileOutputStream outputStream = null;
        String internalStoragePath = null;

        try {
            // 获取 assets 文件夹中的文件
            inputStream = context.getAssets().open(fileName);

            // 创建内部存储目录
            File internalFile = new File(context.getFilesDir(), fileName);
            outputStream = new FileOutputStream(internalFile);

            // 定义缓冲区
            byte[] buffer = new byte[1024];
            int length;

            // 读取和写入文件
            while ((length = inputStream.read(buffer)) > 0) {
                outputStream.write(buffer, 0, length);
            }

            // 文件写入成功，获取文件的完整路径
            internalStoragePath = internalFile.getAbsolutePath();

        } catch (IOException e) {
            e.printStackTrace();
            internalStoragePath = "";  // 写入失败返回空路径

        } finally {
            // 关闭输入输出流
            try {
                if (inputStream != null) {
                    inputStream.close();
                }
                if (outputStream != null) {
                    outputStream.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        return internalStoragePath;
    }
}