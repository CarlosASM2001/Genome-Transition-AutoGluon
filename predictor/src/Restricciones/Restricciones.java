/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Restricciones;

import gene.feature.Exon;
import gene.feature.Information;
import gene.feature.Intron;
import gene.feature.Model;
import gene.information.GeneConstructor;
import java.util.ArrayList;
import java.util.List;
import util.MiddleWare;

/**
 *
 * @author gerson
 */
public class Restricciones {

    /**
     * El metodo que se encarga de revisar si se cumplen las restricciones de
     * tamaño establecidas en Model.java Tanto para exones e intrones
     *
     * @param start La posicion de inicioExon del par de exon, bien es la
     * posicion del AG (+2)
     * @param end La posicion final del par de exon, bien es la posicion del GT
     * (-1) de la posicion siguiente en lista del inicioExon
     * @return True si se cumple la condicion, false al contrario
     */
    public static boolean checkSizeExon(int start, int end) {

        if ((end - start + 1) > Model.minExon && (end - start + 1) < Model.maxExon) {
            return true;
        } else {
            return false;
        }
    }

    public static boolean checkSizeIntron(int start, int end) {

        if ((end - start + 1) > Model.minIntron && ((end - start + 1)) < Model.maxIntron) {
            return true;
        } else {
            return false;
        }
    }

    /**
     * Función para revisar si la longitud de los exones creados son impares y
     * multiplos de 3
     *
     * @param start Posicion inicial de exon a evaluar
     * @param end Posicion final de exon a evaluar
     * @return Verdadero si cumple la restriccion, falso de lo contrario
     */
    public static boolean checkLength(int length) {

        if ((length) % 3 != 0) {
            return false;
        } else {
            return true;
        }

        //System.out.println("El exon con coordenadas(" + start + "," + end + ") " + message + "con la condicion de ser de longitud impar");


    }

    /*  public static boolean checkStops(List <Exon> exons, GeneConstructor gene){

     String aux = gene.getInnerInfo(exons.get(0).getStart(), exons.get(0).getEnd()).toString();
     // Se adecua para evaluación la secuencia del exon
     aux=aux.replace(", ", "");
     aux = aux.substring(1, aux.length()-1);
     if(aux.contains(Model.stops[0]) || aux.contains(Model.stops[1]) || aux.contains(Model.stops[2]))
     return false;
     else
     return true;
     }
     */
    public static boolean checkStops(List<Exon> exons, Integer end) {

        List<Integer> coords = new ArrayList<>();

        boolean pass = true;
        int cantExons = exons.size();
        String orf = "";
        for (int i = 0; i < cantExons; i++) {
            orf = orf + exons.get(i).toString();
            if (i < (cantExons - 1)) {
                if (!(end > exons.get(i).getStart().position && end > exons.get(i).getEnd().position)) {
                    pass = false;
                }
            } else {
                if (!(end > exons.get(i).getStart().position && end >= exons.get(i).getEnd().position)) {
                    pass = false;
                }

            }

        }
        // Se chequea que el orf tenga una sola parada.
        coords = new MiddleWare().getPositionsPatron(orf, false);
        int numCoord = 1, numCoords = coords.size();

        for (Integer coord : coords) {
            if (((coord.intValue() - 2) % 3 == 0) && (numCoord != numCoords)) {
                pass = false;
                break;
            }
            numCoord++;

        }

        return pass;
    }

    public static void addInnerInfo(List<Exon> exons, List<Intron> introns, GeneConstructor gene) throws Exception {

        Information inicioExon, finExon, inicioIntron, finIntron;
        int inicioE, finE, inicioI, finI;

        for (int i = 0; i < exons.size(); i++) {

            Exon exonTemp;

            inicioExon = exons.get(i).getStart();
            inicioE = inicioExon.position;
            finExon = exons.get(i).getEnd();
            finE = finExon.position;
            //inicioExon = gene.getData(inicioE-1);            
            //finExon = gene.getData(finE-1); 

            exonTemp = new Exon(inicioExon, finExon, gene.getInnerInfo(inicioE + 1, finE));

            exons.set(i, new Exon(inicioExon, finExon, gene.getInnerInfo(inicioE + 1, finE)));
        }
        for (int i = 0; i < introns.size(); i++) {

            inicioIntron = introns.get(i).getStart();
            inicioI = inicioIntron.position;
            finIntron = introns.get(i).getEnd();
            finI = finIntron.position;
            //inicioIntron = gene.getData(inicioI - 1);
            //finIntron = gene.getData(finI - 2);
            Intron intronTemp = new Intron(inicioIntron, finIntron, gene.getInnerInfo(inicioI + 1, finI));

            introns.set(i, new Intron(inicioIntron, finIntron, gene.getInnerInfo(inicioI + 1, finI)));
        }

    }
}
