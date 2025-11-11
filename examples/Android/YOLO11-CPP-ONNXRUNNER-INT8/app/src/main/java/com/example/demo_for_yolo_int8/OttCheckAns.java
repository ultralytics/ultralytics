package com.example.demo_for_yolo_int8;

import android.os.Parcel;
import android.os.Parcelable;

public class OttCheckAns implements Parcelable {
    public Point startPoint; // 左上角坐标
    public Point endPoint; // 右下角坐标
    public String boxName; // 分类名称
    public double score; // 置信度

    public OttCheckAns() {
    }

    public OttCheckAns(double startX, double startY, double endX, double endY, String boxName, double score) {
        this.startPoint = new Point();
        this.startPoint.x = startX;
        this.startPoint.y = startY;

        this.endPoint = new Point();
        this.endPoint.x = endX;
        this.endPoint.y = endY;

        this.boxName = boxName;
        this.score = score;
    }

    protected OttCheckAns(Parcel in) {
        startPoint = in.readParcelable(Point.class.getClassLoader());
        endPoint = in.readParcelable(Point.class.getClassLoader());
        boxName = in.readString();
        score = in.readDouble();
    }

    public static final Creator<OttCheckAns> CREATOR = new Creator<OttCheckAns>() {
        @Override
        public OttCheckAns createFromParcel(Parcel in) {
            return new OttCheckAns(in);
        }

        @Override
        public OttCheckAns[] newArray(int size) {
            return new OttCheckAns[size];
        }
    };

    @Override
    public int describeContents() {
        return 0;
    }

    @Override
    public void writeToParcel(Parcel dest, int flags) {
        dest.writeParcelable(startPoint, flags);
        dest.writeParcelable(endPoint, flags);
        dest.writeString(boxName);
        dest.writeDouble(score);
    }
}