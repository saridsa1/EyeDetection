package com.android.noweaj.eyedetection;

import android.content.Context;
import android.content.DialogInterface;
import android.os.AsyncTask;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.w3c.dom.Text;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity implements JavaCameraView.CvCameraViewListener2, View.OnClickListener, SeekBar.OnSeekBarChangeListener, CompoundButton.OnCheckedChangeListener {

    private static final String TAG = MainActivity.class.getSimpleName();

    /**
     * OpenCV
     */

    public static final int JAVA_DETECTOR = 0;
    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);

    private File cascadeFileFace, cascadeFileEye;
    private Mat pMat, pRgba, pGray;
    private CascadeClassifier pJavaDetectorFace, pJavaDetectorEye;
    private CascadeClassifier mJavaDetectorFace, mJavaDetectorEye;
    private int pDetectorType = JAVA_DETECTOR, mDetectorType = JAVA_DETECTOR;
    private String[] pDetectorName, mDetectorName;
    private float pRelativeFaceSize = 0.2f, mRelativeFaceSize = 0.2f;
    private int pAbsoluteFaceSize = 0, mAbsoluteFaceSize = 0;
    private int px, py;
    private Point pPoint;
    private Mat pTemplateR, pTemplateL;

    CameraBridgeViewBase camera;

    CheckBox    cb_grayscale;
    ImageView   iv_eyeLeft, iv_eyeRight;

    SeekBar     sb_brightness, sb_contrast, sb_sharpness, sb_gamma;
    Button      b_brightness_plus, b_brightness_minus, b_contrast_plus, b_contrast_minus,
            b_sharpness_plus, b_sharpness_minus, b_gamma_plus, b_gamma_minus;
    TextView    tv_brightness_value, tv_contrast_value, tv_sharpness_value, tv_gamma_value
            , tv_isdetected_lefteye, tv_isdetected_righteye;

    int brightness = 0, contrast = 0, sharpness = 0, gamma = 0;
    int brightnessMax = 49, contrastMax = 14, sharpnessMax = 149, gammaMax = 15;

    boolean isGrayscale = false;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status){
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully.");

                    try{
                        byte[] buffer = new byte[4096];
                        int bytesRead;

                        // Face
                        InputStream is_face = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDirFace = getDir("cascade", Context.MODE_PRIVATE);
                        cascadeFileFace = new File(cascadeDirFace, "lbpcascade_frontalface.xml");
                        FileOutputStream os_face = new FileOutputStream(cascadeFileFace);

                        while((bytesRead = is_face.read(buffer)) != -1){
                            os_face.write(buffer, 0, bytesRead);
                        }

                        // Eye
                        InputStream is_eye = getResources().openRawResource(R.raw.haarcascade_lefteye_2splits);
                        File cascadeDirEye = getDir("cascade", Context.MODE_PRIVATE);
                        cascadeFileEye = new File(cascadeDirEye, "haarcascade_lefteye_2splits.xml");
                        FileOutputStream os_eye = new FileOutputStream(cascadeFileEye);

                        while((bytesRead = is_eye.read(buffer)) != -1){
                            os_eye.write(buffer, 0, bytesRead);
                        }

                        pJavaDetectorFace = new CascadeClassifier(cascadeFileFace.getAbsolutePath());
                        pJavaDetectorFace.load(cascadeFileFace.getAbsolutePath());
                        if(pJavaDetectorFace.empty()){
                            Log.e(TAG, "Failed to load cascade classifier for pFace");
                            pJavaDetectorFace = null;
                        } else {
                            Log.i(TAG, "Loaded cascade classifier from " + cascadeFileFace.getAbsolutePath());
                        }

                        pJavaDetectorEye = new CascadeClassifier(cascadeFileEye.getAbsolutePath());
                        pJavaDetectorEye.load(cascadeFileEye.getAbsolutePath());
                        if(pJavaDetectorEye.empty()){
                            Log.e(TAG, "Failed to load cascade classifier for pEye");
                            pJavaDetectorEye = null;
                        } else {
                            Log.i(TAG, "Loaded cascade classifier from " + cascadeFileEye.getAbsolutePath());
                        }

                        mJavaDetectorFace = new CascadeClassifier(cascadeFileFace.getAbsolutePath());
                        mJavaDetectorFace.load(cascadeFileFace.getAbsolutePath());
                        if(mJavaDetectorFace.empty()){
                            Log.e(TAG, "Failed to load cascade classifier for mFace");
                            mJavaDetectorFace = null;
                        } else {
                            Log.i(TAG, "Loaded cascade classifier from " + cascadeFileFace.getAbsolutePath());
                        }

                        mJavaDetectorEye = new CascadeClassifier(cascadeFileEye.getAbsolutePath());
                        mJavaDetectorEye.load(cascadeFileEye.getAbsolutePath());
                        if(mJavaDetectorEye.empty()){
                            Log.e(TAG, "Failed to load cascade classifier for mEye");
                            mJavaDetectorEye = null;
                        } else {
                            Log.i(TAG, "Loaded cascade classifier from " + cascadeFileEye.getAbsolutePath());
                        }

                        is_face.close();
                        os_face.close();
                        is_eye.close();
                        os_eye.close();
                        cascadeDirFace.delete();
                        cascadeDirEye.delete();

                    } catch (IOException e){
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                    camera.setCameraIndex(1);
                    camera.enableView();

                }
                break;
                default:
                {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public MainActivity(){
        pDetectorName = new String[2];
        pDetectorName[JAVA_DETECTOR] = "Java";

        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camera = (CameraBridgeViewBase) findViewById(R.id.camera);
        camera.setVisibility(SurfaceView.VISIBLE);
        camera.setCvCameraViewListener(this);
        camera.setOnClickListener(this);

        cb_grayscale = (CheckBox) findViewById(R.id.cb_grayscale);

        cb_grayscale.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if(isChecked){
                    isGrayscale = true;
                } else {
                    isGrayscale = false;
                }
            }
        });

        iv_eyeLeft = (ImageView) findViewById(R.id.iv_eyeLeft);
        iv_eyeRight = (ImageView) findViewById(R.id.iv_eyeRight);

        sb_brightness = (SeekBar) findViewById(R.id.sb_brightness);
        sb_brightness.setMax(brightnessMax);
        sb_brightness.setOnSeekBarChangeListener(this);
        sb_contrast = (SeekBar) findViewById(R.id.sb_contrast);
        sb_contrast.setMax(contrastMax);
        sb_contrast.setOnSeekBarChangeListener(this);
        sb_sharpness = (SeekBar) findViewById(R.id.sb_sharpness);
        sb_sharpness.setMax(sharpnessMax);
        sb_sharpness.setOnSeekBarChangeListener(this);
        sb_gamma = (SeekBar) findViewById(R.id.sb_gamma);
        sb_gamma.setMax(gammaMax);
        sb_gamma.setOnSeekBarChangeListener(this);

        b_brightness_plus = (Button) findViewById(R.id.b_brightness_plus);
        b_brightness_plus.setOnClickListener(this);
        b_brightness_minus = (Button) findViewById(R.id.b_brightness_minus);
        b_brightness_minus.setOnClickListener(this);
        b_contrast_plus = (Button) findViewById(R.id.b_contrast_plus);
        b_contrast_plus.setOnClickListener(this);
        b_contrast_minus = (Button) findViewById(R.id.b_contrast_minus);
        b_contrast_minus.setOnClickListener(this);
        b_sharpness_plus = (Button) findViewById(R.id.b_sharpness_plus);
        b_sharpness_plus.setOnClickListener(this);
        b_sharpness_minus = (Button) findViewById(R.id.b_sharpness_minus);
        b_sharpness_minus.setOnClickListener(this);
        b_gamma_plus = (Button) findViewById(R.id.b_gamma_plus);
        b_gamma_plus.setOnClickListener(this);
        b_gamma_minus = (Button) findViewById(R.id.b_gamma_minus);
        b_gamma_minus.setOnClickListener(this);

        // Pops up dialog to manually put values
        tv_brightness_value = (TextView) findViewById(R.id.tv_brightness_value);
        tv_brightness_value.setOnClickListener(this);
        tv_contrast_value = (TextView) findViewById(R.id.tv_contrast_value);
        tv_contrast_value.setOnClickListener(this);
        tv_sharpness_value = (TextView) findViewById(R.id.tv_sharpness_value);
        tv_sharpness_value.setOnClickListener(this);
        tv_gamma_value = (TextView) findViewById(R.id.tv_gamma_value);
        tv_gamma_value.setOnClickListener(this);

        tv_isdetected_lefteye = (TextView) findViewById(R.id.tv_isdetected_lefteye);
        tv_isdetected_righteye = (TextView) findViewById(R.id.tv_isdetected_righteye);
    }

    @Override
    public void onPause(){
        super.onPause();
        if(camera != null){
            camera.disableView();
        }
    }

    @Override
    public void onResume(){
        super.onResume();
        if(!OpenCVLoader.initDebug()){
            Log.d(TAG, "Internal OpenCV library not found");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onDestroy(){
        super.onDestroy();
        if(camera != null){
            camera.disableView();
        }
    }

    @Override
    public void onBackPressed() {
        new AlertDialog.Builder(this)
                .setTitle(R.string.alert_title)
                .setMessage(R.string.alert_message)
                .setPositiveButton("YES", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {

                        if(camera != null){
                            camera.disableView();
                        }
                        finish();
                    }
                })
                .setNegativeButton("NO", null).show();
    }

    @Override
    public void onClick(View v){
        switch (v.getId()){
            case R.id.b_brightness_plus:
                if(brightness < brightnessMax){
                    brightness++;
                }
                break;
            case R.id.b_brightness_minus:
                if(brightness > 1){
                    brightness--;
                }
                break;
            case R.id.b_contrast_plus:
                if(contrast < contrastMax){
                    contrast++;
                }
                break;
            case R.id.b_contrast_minus:
                if(contrast > 1){
                    contrast--;
                }
                break;
            case R.id.b_sharpness_plus:
                if(sharpness < sharpnessMax){
                    sharpness++;
                }
                break;
            case R.id.b_sharpness_minus:
                if (sharpness > 1){
                    sharpness--;
                }
                break;
            case R.id.b_gamma_plus:
                if(gamma < gammaMax){
                    gamma++;
                }
                break;
            case R.id.b_gamma_minus:
                if(gamma > 1){
                    gamma--;
                }
                break;
            case R.id.tv_brightness_value:
                break;
            case R.id.tv_contrast_value:
                break;
            case R.id.tv_sharpness_value:
                break;
            case R.id.tv_gamma_value:
                break;
            case R.id.camera:

                break;
            default:
                break;
        }
        updateValues();
    }

    private void updateValues(){
        tv_brightness_value.setText(brightness);
        tv_contrast_value.setText(contrast);
        tv_sharpness_value.setText(sharpness);
        tv_gamma_value.setText(gamma);
    }

    @Override
    public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
        switch(seekBar.getId()){
            case R.id.sb_brightness:
                brightness = progress;
                break;
            case R.id.sb_contrast:
                contrast = progress;
                break;
            case R.id.sb_sharpness:
                sharpness = progress;
                break;
            case R.id.sb_gamma:
                gamma = progress;
                break;
            default:
                break;
        }
    }

    @Override
    public void onStartTrackingTouch(SeekBar seekBar) {

    }

    @Override
    public void onStopTrackingTouch(SeekBar seekBar) {

    }

    @Override
    public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
        if(isChecked){
            isGrayscale = true;
        } else {
            isGrayscale = false;
        }
    }

    /**
     *
     */
    class DoAsyncTask extends AsyncTask<Integer, Integer, Long>{

        @Override
        protected void onCancelled(){
            super.onCancelled();
        }

        @Override
        protected void onPostExecute(Long result){

        }

        @Override
        protected void onPreExecute(){

        }

        @Override
        protected void onProgressUpdate(Integer ... values){

        }

        @Override
        protected Long doInBackground(Integer... integers) {
            return null;
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        pMat = new Mat();
        pRgba = new Mat();
        pGray = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        pMat.release();
        pRgba.release();
        pGray.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        if(isGrayscale){
            pRgba = inputFrame.gray();
        } else {
            pRgba = inputFrame.rgba();
        }

        pGray = inputFrame.gray();

        Mat modifyRgba = Imgproc.getRotationMatrix2D(new Point(pRgba.cols()/2, pRgba.rows()/2), 90, 1);
        Mat modifyGray = Imgproc.getRotationMatrix2D(new Point(pGray.cols()/2, pGray.rows()/2), 90, 1);
        Imgproc.warpAffine(pRgba, pRgba, modifyRgba, pRgba.size());
        Imgproc.warpAffine(pGray, pGray, modifyGray, pGray.size());

        if(pAbsoluteFaceSize == 0){
            int height = pGray.rows();
            if(Math.round(height * pRelativeFaceSize) > 0){
                pAbsoluteFaceSize = Math.round(height * pRelativeFaceSize);
            }
        }

        MatOfRect faces = new MatOfRect();

        if(pDetectorType == JAVA_DETECTOR){
            if(pJavaDetectorFace != null){
                pJavaDetectorFace.detectMultiScale(pGray, faces, 1.1, 2, 2, new Size(pAbsoluteFaceSize, pAbsoluteFaceSize), new Size());
            }
        } else {
            Log.e(TAG, "Detection method is not selected");
        }

        Rect[] facesArray = faces.toArray();
        for(int i=0; i<facesArray.length; i++){
            px = (int) ((facesArray[i].br().x - facesArray[i].tl().x)/2 + facesArray[i].tl().x);
            py = (int) ((facesArray[i].br().y - facesArray[i].tl().y)/2 + facesArray[i].tl().y);
            pPoint = new Point(px, py);

            Imgproc.circle(pRgba, pPoint, 2 * (facesArray[i].width/3), FACE_RECT_COLOR, 3);

            Rect r = facesArray[i];
            Rect eyearea_right = new Rect(r.x + r.width / 16, (int) (r.y + (r.height / 4.5)), (r.width - 2 * r.width / 16) / 2, (int) (r.height / 3.0));
            Rect eyearea_left = new Rect(r.x + r.width / 16 + (r.width - 2 * r.width / 16) / 2, (int) (r.y + (r.height / 4.5)), (r.width - 2 * r.width / 16) / 2, (int) (r.height / 3.0));

            Imgproc.rectangle(pRgba, eyearea_left.tl(), eyearea_left.br(), new Scalar(255, 0, 0, 255), 2);
            Imgproc.rectangle(pRgba, eyearea_right.tl(), eyearea_right.br(), new Scalar(255, 0, 0, 255), 2);

            //Mat interestMat =

            pTemplateR = get_pTemplate(pJavaDetectorEye, eyearea_right, 24);
            pTemplateL = get_pTemplate(pJavaDetectorEye, eyearea_left, 24);
        }

        //Imgproc.ellipse(pRgba, new Point(pRgba.cols()/2, pRgba.rows()/2), new Size(90, 120), 180, 0, 360, new Scalar(255, 255, 0, 255), 3, 8, 0);

        return pRgba;
    }

    private Mat get_pTemplate(CascadeClassifier pClassifier, Rect pArea, int pSize){
        Mat template = new Mat();
        Mat mROI = pGray.submat(pArea);
        MatOfRect eyes = new MatOfRect();
        Point iris = new Point();
        Rect eye_template = new Rect();

        pClassifier.detectMultiScale(mROI, eyes, 1.15, 2, Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_SCALE_IMAGE, new Size(30, 30), new Size());

        Rect[] eyesArray = eyes.toArray();
        for(int i=0; i<eyesArray.length;){
            Rect e = eyesArray[i];
            e.x = pArea.x + e.x;
            e.y = pArea.y + e.y;
            Rect eye_only_rectangle = new Rect((int) e.tl().x, (int) (e.tl().y + e.height * 0.4),
                    (int) e.width, (int) (e.height * 0.6));
            mROI = pGray.submat(eye_only_rectangle);
            Mat vyrez = pGray.submat(eye_only_rectangle);

            Core.MinMaxLocResult mmg = Core.minMaxLoc(mROI);

            Imgproc.circle(vyrez, mmg.minLoc, 2, new Scalar(255,255,255,255), 2);
            iris.x = mmg.minLoc.x + eye_only_rectangle.x;
            iris.y = mmg.minLoc.y + eye_only_rectangle.y;

            eye_template = new Rect((int) iris.x - pSize / 2, (int) iris.y - pSize / 2, pSize, pSize);
            //Imgproc.rectangle(pGray, eye_template.tl(), eye_template.br(), new Scalar(255,255,255,255), 2);
            Imgproc.rectangle(pRgba, eye_template.tl(), eye_template.br(), new Scalar(255,255,255,255), 2);
            template = (pGray.submat(eye_template)).clone();
            return template;
        }
        return template;
    }

}
