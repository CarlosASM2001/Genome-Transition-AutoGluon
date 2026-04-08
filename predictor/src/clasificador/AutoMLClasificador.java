/*
 * AutoMLClasificador.java
 * Cliente HTTP para integrar con el servicio Genome Transition Auto ML (FastAPI)
 *
 * Este clasificador realiza peticiones REST al servicio de AutoGluon para predecir
 * zonas de transición genómica (EI, IE, ZE, EZ) usando modelos de Machine Learning.
 *
 * Copyright (C) 2024
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 */

package clasificador;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import org.json.JSONArray;
import org.json.JSONObject;

/**
 * Clasificador que utiliza el servicio REST de AutoML basado en AutoGluon
 */
public class AutoMLClasificador {

    private String apiUrl;


    public AutoMLClasificador() {
        this.apiUrl = "http://143.198.77.77:8000";
    }

    /**
     * Realiza una predicción de todas las zonas de transición genómica.
     * Envía únicamente la secuencia al servicio, sin parámetros adicionales.
     *
     * @param secuencia Secuencia nucleotídica (ATGC)
     * @return Objeto AutoMLResult con las predicciones de todas las zonas
     * @throws Exception Si hay error en la comunicación o predicción
     */
    public AutoMLResult predict(String secuencia) throws Exception {

        // Construir el JSON de la petición (solo la secuencia)
        JSONObject requestBody = new JSONObject();
        requestBody.put("sequence", secuencia.toLowerCase());

        // Realizar la petición HTTP POST
        String responseJson = sendPostRequest(apiUrl + "/predict", requestBody.toString());

        // Parsear la respuesta
        JSONObject response = new JSONObject(responseJson);

        // Extraer las listas de posiciones
        List<Integer> eiPositions = jsonArrayToIntegerList(response.getJSONArray("ei"));
        List<Integer> iePositions = jsonArrayToIntegerList(response.getJSONArray("ie"));
        List<Integer> zePositions = jsonArrayToIntegerList(response.getJSONArray("ze"));
        List<Integer> ezPositions = jsonArrayToIntegerList(response.getJSONArray("ez"));

        return new AutoMLResult(eiPositions, iePositions, zePositions, ezPositions);
    }

    /**
     * Envía una petición POST al servicio.
     *
     * @param urlString URL del endpoint
     * @param jsonBody Cuerpo JSON de la petición
     * @return Respuesta del servidor como String
     * @throws Exception Si hay error en la comunicación
     */
    private String sendPostRequest(String urlString, String jsonBody) throws Exception {
        URL url = new URL(urlString);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();

        try {
            // Configurar la conexión
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setRequestProperty("Accept", "application/json");
            conn.setDoOutput(true);
            conn.setConnectTimeout(60000);
            conn.setReadTimeout(60000);

            // Enviar el cuerpo de la petición
            try (OutputStream os = conn.getOutputStream()) {
                byte[] input = jsonBody.getBytes(StandardCharsets.UTF_8);
                os.write(input, 0, input.length);
            }

            // Leer la respuesta
            int responseCode = conn.getResponseCode();

            if (responseCode == 200) {
                // Respuesta exitosa
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
                    StringBuilder response = new StringBuilder();
                    String responseLine;
                    while ((responseLine = br.readLine()) != null) {
                        response.append(responseLine.trim());
                    }
                    return response.toString();
                }
            } else {
                // Error en la petición
                String errorMessage;
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getErrorStream(), StandardCharsets.UTF_8))) {
                    StringBuilder error = new StringBuilder();
                    String errorLine;
                    while ((errorLine = br.readLine()) != null) {
                        error.append(errorLine.trim());
                    }
                    errorMessage = error.toString();
                }
                throw new Exception("Error en petición AutoML (código " + responseCode + "): " + errorMessage);
            }
        } finally {
            conn.disconnect();
        }
    }

    /**
     * Convierte un JSONArray a una lista de enteros.
     *
     * @param jsonArray Array JSON con enteros
     * @return Lista de Integer
     */
    private List<Integer> jsonArrayToIntegerList(JSONArray jsonArray) {
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < jsonArray.length(); i++) {
            list.add(jsonArray.getInt(i));
        }
        return list;
    }

    /**
     * Clase interna para almacenar los resultados de una predicción completa.
     */
    public static class AutoMLResult {
        private List<Integer> eiPositions;
        private List<Integer> iePositions;
        private List<Integer> zePositions;
        private List<Integer> ezPositions;

        public AutoMLResult(List<Integer> ei, List<Integer> ie, List<Integer> ze, List<Integer> ez) {
            this.eiPositions = ei;
            this.iePositions = ie;
            this.zePositions = ze;
            this.ezPositions = ez;
        }

        public List<Integer> getEiPositions() { return eiPositions; }
        public List<Integer> getIePositions() { return iePositions; }
        public List<Integer> getZePositions() { return zePositions; }
        public List<Integer> getEzPositions() { return ezPositions; }

        @Override
        public String toString() {
            return String.format("AutoMLResult{EI=%d, IE=%d, ZE=%d, EZ=%d}",
                    eiPositions.size(), iePositions.size(), zePositions.size(), ezPositions.size());
        }
    }
}
