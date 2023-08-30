package com.example.fashion_mnist

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.ColorSpace
import android.graphics.ImageFormat
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import com.example.fashion_mnist.ml.FashionMnistRescaling
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.TransformToGrayscaleOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {

    val PICK_IMAGE_REQUEST = 1 // константа для кода запроса изображения
    lateinit var select_btn: Button
    lateinit var predict_btn: Button
    lateinit var pred_img: ImageView
    lateinit var pred_text: TextView
    lateinit var bitmap: Bitmap


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        select_btn = findViewById(R.id.select_1)
        predict_btn = findViewById(R.id.predict_1)
        pred_img = findViewById(R.id.image_1)
        pred_text = findViewById(R.id.text_1)



        select_btn.setOnClickListener {
            // создаем новое намерение для выбора изображения из галереи
            val intent = Intent(Intent.ACTION_PICK)
            intent.type = "image/*"
            // запускаем активность для получения результата
            startActivityForResult(intent, PICK_IMAGE_REQUEST)
        }

        predict_btn.setOnClickListener {
            // Доп. обработка изображений
            val imgProc = ImageProcessor.Builder()
                .add(ResizeOp(28, 28, ResizeOp.ResizeMethod.BILINEAR))
                .add(TransformToGrayscaleOp()).build()


            var img = TensorImage(DataType.FLOAT32)
            img.load(bitmap)


            img = imgProc.process(img)

            // Посмотреть размерность
//            val tensor = img.tensorBuffer
//            val shape = tensor.shape
//            pred_text.setText("(${shape[0]}, ${shape[1]}, ${shape[2]}, ${shape[3]})")





            val model = FashionMnistRescaling.newInstance(this)
//
//
//            // Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 28, 28, 1), DataType.FLOAT32)
            inputFeature0.loadBuffer(img.buffer)
//
//            // Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray


            var maxIdx = 0
            outputFeature0.forEachIndexed { index, fl ->
                if (outputFeature0[maxIdx] > fl) {
                    maxIdx = index
                }
            }

            val label_1 = application.assets.open("Labels.txt").bufferedReader().readLines()
////             Преобразуем массив в список значений
            val values = outputFeature0.map { it }

            // Находим максимальное значение
            val maxValue = values.maxOrNull()

            // Получаем его индекс
            val maxIndex = values.indexOf(maxValue)

//             Выводим индекс
            pred_text.setText(label_1[maxIndex])



            model.close()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {

        super.onActivityResult(requestCode, resultCode, data)
        // проверяем, что результат соответствует нашему коду запроса и не пустой
        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null) {
            // получаем URI выбранного изображения
            val imageUri = data.data;
            // устанавливаем его в наш ImageView
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, imageUri)
            pred_img.setImageBitmap(bitmap)

        }
    }
}