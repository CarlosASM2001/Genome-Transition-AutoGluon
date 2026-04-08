/*
    PreprocesadorUTR.java
    Prepara datos para experimentos ILP cajas promotoras usando ejemplos desde EPD.

    Copyright (C) 2016 
    Yackson Ramirez (yacson.ramirez@gmail.com), Jose Lopez (jlopez@unet.edu.ve.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>

    */

/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Vector;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 *
 * @author Yackson Ramirez (yacson.ramirez@gmail.com), Jose Lopez (jlopez@unet.edu.ve.
 */
public class PreprocesadorUTR {

    public static void main(String[] args) throws FileNotFoundException, IOException {
        PreprocesadorUTR prePro = new PreprocesadorUTR();
        //prePro.generadorPromots("promotores_Inr_small.txt", "promtsMuestrados.txt", "Inr", 5);
        prePro.generadorPromots("promotores_Inr.txt", "promtsMuestrados.txt", "Inr", 10);

    }

    public void generadorPromots(String promtsFasta, String promtsMuestrados, String consSearch, int cantPromotores) throws FileNotFoundException, IOException, StringIndexOutOfBoundsException {

        String consensus = "";

        if (consSearch.equals("CAAT")) {// t/c/a	c/t	t/c	A/g	g/a	C	C	A	a/t	T/a	c/g	A/g
            consensus = "[TCA][CT][TC][AG][GA]CCA[AT][TA][CG][AG]";
        }

        if (consSearch.equals("GC")) { //a/t	a/g g/t/a G/a G	G C/t/a G/a G/t	g/a/t g/t c/t t/c t/g
            consensus = "[AT][AG][GTA][GA]GG[CTA][GA][GT][GAT][GT][CT][TC][TG]";
        }

        if (consSearch.equals("TATA")) { // g/c	t	A/t	T	A/t	A/T	A	a/t	g/a	g/c	c/g	g/c	g/c	g/c	g/c
            consensus = "[GC]T[AT]T[AT][AT]A[AT][GA][GC][CG][GC][GC][GC][GC]";
        }

        if (consSearch.equals("Inr")) {// t/g	C	A/t	g/t/c	T/c/a	c/t	t/c/g	t/c
            consensus = "[TG]C[AT][GTC][TCA][CT][TCG][TC]";
        }

        if (consSearch.equals("DPE")) {
            consensus = "[AG]G[AT][CT][GAC]";
        }

        // hasConsense = Pattern.matches("[TC][TC]A[ACGT][TA][TC][TC]", motif);// Consenso Inr

        String promotorParcial, promotor = "";
        String esc = ">";

        File promotsFasta = new File(promtsFasta); // Los promotores como vienen desde EPD.
        File promotsFormat = new File("ejemplosPromts.txt"); // Todos los promotores ya formateados para muestrear
        File promotsConsensus = new File("ejemplosPromtsConsensus.txt"); // Todos los promotores con consenso consSearch.
        File promotsMuestreo = new File(promtsMuestrados); // Los promotores muestrados desde los cuales extraer ejemplos para 
                                                           // experimento ILP.

        BufferedReader promotoresFasta = new BufferedReader(new FileReader(promotsFasta));

        FileWriter ejemplosPromtsFormat = new FileWriter(promotsFormat); 
        FileWriter ejemplosPromtsConsensus = new FileWriter(promotsConsensus); // Aqui se guardan  los promotores
        // que si' tienen el consenso consSearch.

        String[] fastaHeader;
        promotorParcial = promotoresFasta.readLine();
        fastaHeader = promotorParcial.split(" ");
        String promoterID = fastaHeader[1];
        int contPromGlob = 0;

        boolean escape = false;

        while (promotoresFasta.ready()) {

            promotorParcial = promotoresFasta.readLine();

            do {

                promotor = promotor + promotorParcial;
                if (promotoresFasta.ready()) {
                    promotorParcial = promotoresFasta.readLine();
                    escape = promotorParcial.startsWith(esc);
                } else {
                    escape = true;
                }

            } while (!escape);

            if (promotoresFasta.ready()) {
                ejemplosPromtsFormat.write(promoterID + " " + promotor + "\n");
            } else {
                ejemplosPromtsFormat.write(promoterID + " " + promotor);
            }
            if (promotoresFasta.ready()) {
                fastaHeader = promotorParcial.split(" ");
                promoterID = fastaHeader[1];
            }

            promotor = "";
            contPromGlob++;

        }
        ejemplosPromtsFormat.close();
        BufferedReader promotores = new BufferedReader(new FileReader(promotsFormat));
        int cantPromots = 0;
        String promConConsensus, temporal;
        boolean hasConsense = false;
        List<Integer> posConsensus = new ArrayList<>();

        temporal = promotores.readLine();
        int posMatched, posTestCons = 48;

        while (promotores.ready()) { // Se derminan los promotores que tienen el consenso en proceso.

            promConConsensus = temporal.split(" ")[1];
            Pattern p = Pattern.compile(consensus);
            Matcher m = p.matcher(promConConsensus);

            if (consSearch.equals("Inr")) {
                //
                while (m.find()) {
                    posMatched = m.start();
                    posConsensus.add(posMatched);
                }
                if (!posConsensus.isEmpty()) {
                    hasConsense = posConsensus.contains(posTestCons);
                    posConsensus.clear();
                }
            } else {
                hasConsense = m.find();
            }

            if (hasConsense && promotores.ready()) {
                ejemplosPromtsConsensus.write(temporal + "\n");
                cantPromots++;
            } else {
                if (hasConsense && !promotores.ready()) {
                    ejemplosPromtsConsensus.write(temporal);
                    cantPromots++;
                }
            }

            temporal = promotores.readLine();

            if (!promotores.ready()) {
                promConConsensus = temporal.split(" ")[1];
                p = Pattern.compile(consensus);
                m = p.matcher(promConConsensus);
                
                if (consSearch.equals("Inr")) {
                    //
                    while (m.find()) {
                        posMatched = m.start();
                        posConsensus.add(posMatched);
                    }
                    if (!posConsensus.isEmpty()) {
                        hasConsense = posConsensus.contains(posTestCons);
                        posConsensus.clear();
                    }
                } else {
                    hasConsense = m.find();
                }
                
                if (hasConsense) {
                    ejemplosPromtsConsensus.write(temporal);
                    cantPromots++;
                }
            }
            hasConsense = false;

        }
        ejemplosPromtsConsensus.close();

        promotores.close();
        System.out.println(" Existen " + cantPromots + "/" + contPromGlob + " promotores tipo " + consSearch);

        promotores = new BufferedReader(new FileReader(promotsConsensus));

        FileWriter promtsM = new FileWriter(promotsMuestreo);
        ArrayList<Integer> promotsMuestrear = new ArrayList<>();
        int muestrados = 0;

        if (cantPromotores <= cantPromots) {

            float reference = (float) Math.random() * cantPromots;
            int aMuestrear = Math.round(reference);
            do {
                if ((aMuestrear != 0) && !promotsMuestrear.contains(aMuestrear)) {
                    promotsMuestrear.add(aMuestrear);
                    muestrados++;
                }
                reference = (float) Math.random() * cantPromots;
                aMuestrear = Math.round(reference);

            } while (muestrados < cantPromotores);

            Collections.sort(promotsMuestrear);

            int contMuestreo = 1, muestreados = 0;
            String promot;


            if (promotores.ready()) {

                for (Integer muestra : promotsMuestrear) {

                    do {
                        try {
                            if (promotores.ready()) {
                                promot = promotores.readLine();
                                if (contMuestreo == muestra) {
                                    promtsM.write(promot);
                                    muestreados++;
                                    if (muestreados != promotsMuestrear.size()) {
                                        promtsM.write("\n");
                                    }
                                }
                            } else {
                                System.out.println("Error: No hay la cantidad de promotores que se requieren en el muestreo. Revisar codigo!");
                                System.exit(0);

                            }
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                        contMuestreo++;
                    } while (contMuestreo <= muestra);
                }
            }
        } else {

            System.out.println("No hay esa cantidad de promotores disponibles");

        }
        promtsM.close();
        promotores.close();


    }

    private int printBC(String archivo, Vector eventos) throws IOException {

        FileWriter fichero = new FileWriter(archivo);
        PrintWriter pw = new PrintWriter(fichero);
        int cant_eventos = eventos.size();
        String evento;
        pw.println("base([");

        for (int i = 0; i < cant_eventos; i++) {
            if (i != (cant_eventos - 1)) {
                evento = (String) eventos.elementAt(i) + ",";

            } else {
                evento = (String) eventos.elementAt(i);
            }

            pw.println(evento);
            // System.out.println(evento);
        }

        pw.println("]).");
        pw.close();
        fichero.close();

        return cant_eventos;
    }
}
