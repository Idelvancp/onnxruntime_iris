package com.example.onnxruntime;

import io.flutter.embedding.android.FlutterActivity;
import io.flutter.embedding.engine.FlutterEngine;
import io.flutter.plugin.common.MethodChannel;
import androidx.annotation.NonNull;
import android.util.Log;

import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;

public class MainActivity extends FlutterActivity {
    private static final String CHANNEL = "com.example.audio/audio_processor";

    @Override
    public void configureFlutterEngine(@NonNull FlutterEngine flutterEngine) {
        super.configureFlutterEngine(flutterEngine);

        new MethodChannel(flutterEngine.getDartExecutor().getBinaryMessenger(), CHANNEL)
                .setMethodCallHandler(
                        (call, result) -> {
                            try {
                                switch (call.method) {
                                    case "onnxr":
                                        // Configurar o ambiente ONNX
                                        OrtEnvironment ortEnvironment = OrtEnvironment.getEnvironment();
                                        InputStream inputStream = getResources().openRawResource(R.raw.rf_iris);
                                        byte[] modelBytes = new byte[inputStream.available()];
                                        inputStream.read(modelBytes);
                                        inputStream.close();
                                        OrtSession ortSession = ortEnvironment.createSession(modelBytes);

                                        // Dados de entrada para predição
                                        float[][] inputData = {{6.9f, 3.1f, 5.4f, 2.1f}}; // Exemplo de uma flor `virginica`
                                        FloatBuffer inputBuffer = FloatBuffer.allocate(4);
                                        for (float val : inputData[0]) {
                                            inputBuffer.put(val);
                                        }
                                        inputBuffer.rewind();

                                        // Criar tensor de entrada
                                        OnnxTensor tensor = OnnxTensor.createTensor(ortEnvironment, inputBuffer, new long[]{1, 4});

                                        // Fazer a predição
                                        String inputName = ortSession.getInputNames().iterator().next(); // Nome da entrada
                                        Result output = ortSession.run(Map.of(inputName, tensor));

                                        // Extração dos valores previstos
                                        long[] predictions = (long[]) output.get(0).getValue();
                                       // System.out.println("Predições : " + predictions);
                                        System.out.println("Predições : " + java.util.Arrays.toString(predictions));
                                        // Retornar o resultado
                                        result.success(predictions);
                                        break;

                                    default:
                                        result.notImplemented();
                                        break;
                                }
                            } catch (Exception e) {
                                Log.e("MainActivity", "Erro ao executar tarefa", e);
                                result.error("TASK_ERROR", "Erro ao executar tarefa", e.getMessage());
                            }
                        }
                );
    }
}
