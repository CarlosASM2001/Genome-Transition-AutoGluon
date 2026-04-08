/*
 Analizer.java
 Clase enfocada en el analisis de un objeto GeneConstructor para la formacion
 de lecturas(genes) validos, que son asociadas a una lista llamada "lectures"
 Copyright (C) 2016. 
 Morales Yonathan (yonathan.morales@unet.edu.ve), Jose Lopez (jlopez@unet.edu.ve).

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
package gene.information;

import Libreria.Utilities;
import Restricciones.Restricciones;
import gene.feature.Exon;
import gene.feature.Gene;
import gene.feature.Information;
import gene.feature.Intron;
import gene.feature.Model;
import gene.feature.UTR3p;
import gene.feature.UTR5p;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import pipeline.BioPattern;
//import pipeline.Factor_Transcripcion;
import pipeline.Motivo;
import pipeline.Region;
import estructura.factorTranscripcion;
import util.GenInformation;
import util.MiddleWare;

/**
 * Clase enfocada en el analisis de un objeto GeneConstructor para la formacion
 * de lecturas(genes) validos, que son asociadas a una lista llamada "lectures"
 */
public class Analizer {
    //---------------------------Private Attributes-----------------------------
    // <editor-fold desc="Private Attributes">
    //---tanto el constructor como lectures van a ser utilizados para poder generar las lecturas validas--

    private GeneConstructor constructor;
    //-------- esta lista es donde van ser almacenadas las lecturas----------------
    private List<Gene> lectures;
    //  </editor-fold>
    //---------------------------Constructors-----------------------------------
    // <editor-fold defaultstate="collapsed" desc="Constructors">

    public Analizer() {
        this.lectures = new ArrayList<>();
    }

    //---------------------------------------
    //  </editor-fold>    
    //---------------------------Getters---------------------------------------- 
    // <editor-fold defaultstate="collapsed" desc="Getters">
    public List<Gene> getLectures() {
        return lectures;
    }

    //---------------------------------------
    public GeneConstructor getConstructor() {
        return constructor;
    }

    //  </editor-fold>
    //---------------------------Public Methods--------------------------------- 
    // <editor-fold defaultstate="collapsed" desc="Public Methods">
    /**
     * Metodo usado para instanciar el GeneConstructor a traves del constructor
     * que recibe por parametros el Middle
     *
     * @param middleWare Objeto MiddleWare
     * @param ilpClasificador true para usar ILP, false para usar clasificadores
     * @param useAutoML true para usar AutoML, false para usar Weka
     * @param data Datos del gen
     * @param rutaSecuencia Ruta a la secuencia
     * @param secuencia Secuencia nucleotídica
     */
    public void readFromMiddleWare(MiddleWare middleWare, boolean ilpClasificador, boolean useAutoML, List<String> data, String rutaSecuencia, String secuencia) throws Exception {
        this.constructor = new GeneConstructor(middleWare, ilpClasificador, useAutoML, data, rutaSecuencia, secuencia);
    }

    //---------------------------------------
    /**
     * Metodo usado para instanciar el GeneConstructor a partir de las listas
     * que se estan recibiendo, donde la data es una simple lista de String y
     * las demas son listas de Integer
     */
    public void readFromLists(List<Integer> atg, List<Integer> gt, List<Integer> ag, List<Integer> stops, List<Integer> tss, List<Integer> tts, List<String> data) throws Exception {
        this.constructor = new GeneConstructor();
        this.constructor.initLists(atg, gt, ag, stops, tss, tts, data);
        this.constructor.isCompatibleGene();
    }

    //---------------------------------------
    /**
     * Metodo usado para la construccion de lecturas, al llamar este metodo, se
     * analizaran las listas del GeneConstructor y se llenara la lista de
     * genes(lectures) con todas las lecturas validas, este metodo recibe un
     * boolean que indica si las combinaciones se haran usando un metodo
     * recursivo (true) o iterativo (false)
     */
    public void constructLecturesRes(boolean recursively) throws Exception {
        if (constructor.isCompatibleGene()) {
            int last = constructor.lastData();

            if ((constructor.getGt().isEmpty() || constructor.getAg().isEmpty())) {
                //Caso especial cuando el gen completo es un solo exon
                if (last >= Model.minExon && last <= Model.maxExon) {
                    Information start = constructor.getData(0);
                    Information end = constructor.getData(last);

                    Gene gene = new Gene(start, end);
                    gene.addExon(new Exon(start, end, constructor.getInnerInfo(1, last - 1)));

                    this.lectures.add(gene);
                } else {
                    throw new Exception("La secuencia no contiene exones");
                }
            } else {
                //en este punto, las 4 listas estan llenas y debo hacer las iteraciones
                Gene possibilities = getPosibilities(); // Se generan todos los intrones posibles y se guardan en un gen temporal.

                if (!possibilities.getIntrons().isEmpty()) {
                    ArrayDeque<ArrayDeque<Intron>> mixedIntrons; // Se crea una cola que contendra colas de intrones y cada cola 
                    // correspondera a una lectura posible expresada como secuencia de intrones..

                    if (recursively) {
                        mixedIntrons = this.recursivelyMix(possibilities);
                    } else {
                        mixedIntrons = this.iterativeMix(possibilities);
                    }

                    if (!mixedIntrons.isEmpty()) {
                        this.generateLectures(mixedIntrons);
                    } else {
                        System.out.println("La secuencia no contiene Intrones, no hay lecturas que reportar");
                    }

                } else {
                    throw new Exception("La secuencia que se intenta analizar no contiene intrones y no es un unico exon");
                }
            }
        } else {
            throw new Exception("La secuencia que se intenta analizar no contiene suficientes coordenadas");
        }
    }

    //---------------------------------------
    /**
     * Metodo usado para la construccion de lecturas, al llamar este metodo, se
     * analizaran las listas del GeneConstructor y se llenara la lista de
     * genes(lectures) con todas las lecturas validas, este metodo recibe un
     * boolean que indica si las combinaciones se haran usando un metodo
     * recursivo (true) o iterativo (false)
     */
    public void constructLectures(boolean recursively) throws Exception {
        if (constructor.isCompatibleGene()) {
            int last = constructor.lastData();

            if ((constructor.getGt().isEmpty() || constructor.getAg().isEmpty())) {
                //Caso especial cuando el gen completo es un solo exon
                if (last >= Model.minExon && last <= Model.maxExon) {
                    // Information start = constructor.getData(0);
                    //  Information end = constructor.getData(last);

                    Information start = constructor.getData(0);
                    Information end = constructor.getData(last);

                    Gene gene = new Gene(start, end);
                    gene.addExon(new Exon(start, end));

                    this.lectures.add(gene);
                } else {
                    throw new Exception("El gen que se intenta analizar es incompatible");
                }
            } else {
                //en este punto, las 4 listas estan llenas y debo hacer las iteraciones
                Gene possibilities = getPosibilities();

                if (!possibilities.getIntrons().isEmpty()) {
                    ArrayDeque<ArrayDeque<Intron>> mixedIntrons;

                    if (recursively) {
                        mixedIntrons = this.recursivelyMix(possibilities);
                    } else {
                        mixedIntrons = this.iterativeMix(possibilities);
                    }

                    this.generateLectures(mixedIntrons);
                } else {
                    throw new Exception("El gen que se intenta analizar es incompatible");
                }
            }
        } else {
            throw new Exception("El gen que se intenta analizar es incompatible");
        }

        // Se ajustan las coordenadas para efectos de reporte.
    }

    //---------------------------------------
    /**
     * Metodo usado para instanciar el GeneConstructor a partir de las listas
     * que se estan recibiendo, donde la data es una simple lista de String y
     * las demas son listas de Integer
     *
     * @throws java.lang.Exception
     */
    public Integer definirTSS(Motivo motivoSup, Integer inicioATG, GeneConstructor constructor) throws Exception {

        Integer TSS = 0;
        int barrido = 0, distMin, distMax;
        boolean exploracion = true;
        double mayorProbTSS = 0.0;
        List<Integer> tss = constructor.getTss();
        ArrayList<Double> distPosTss = constructor.getDistPosTss();

        int motivoSupCoordenadas = 0;

        if (motivoSup != null) {

            motivoSupCoordenadas = motivoSup.getCoordenadas()[1];
            String core = motivoSup.getCore();

            switch (core.toLowerCase()) {

                case "caat":
                    distMin = 120;
                    distMax = 140;
                    break;

                case "gc":
                    distMin = 120;
                    distMax = 140;
                    break;

                case "bre":
                    distMin = 30;
                    distMax = 500;
                    break;

                case "tata":
                    distMin = 20;
                    distMax = 40;
                    break;

                default:
                    distMin = 0;
                    distMax = inicioATG - Model.minUTR5p;
                    break;
            }

        } else {

            motivoSupCoordenadas = inicioATG - Model.maxUTR5p;

            if (motivoSupCoordenadas < 0) {

                motivoSupCoordenadas = 0;
                distMin = 0;
                distMax = inicioATG - Model.minUTR5p;

            } else {

                distMin = 0;
                distMax = inicioATG - Model.minUTR5p;

            }

        }

        for (Integer posicion : constructor.getTss()) {

            if ((posicion < (motivoSupCoordenadas + distMin))) {
                barrido++;
            } else {
                break;
            }
        }

        while (exploracion) {

            if (barrido < tss.size()) {

                if ((tss.get(barrido) <= (motivoSupCoordenadas + distMax)) && (inicioATG - tss.get(barrido) >= Model.minUTR5p)) {

                    if (distPosTss.get(barrido) > mayorProbTSS) {
                        mayorProbTSS = distPosTss.get(barrido);
                        TSS = tss.get(barrido);
                    }

                    barrido++;

                } else {
                    exploracion = false;
                }

            } else {
                exploracion = false;
            }

        }

        return TSS;
    }

    /**
     * Metodo usado para la construccion de Regiones 5' para transcrito en
     * proceso. Ya se tiene un ORF y se desea completar el transcrito
     * correspondiente.
     */
    public boolean constructRegionUTR5p(Utilities metaData, GenInformation genInformation, boolean inrILP, boolean consensos) throws Exception {

        boolean utr5pDefined = false;
        ArrayList<Integer> coordsTSS = new ArrayList<>();

        for (Gene lectura : lectures) { // Cada lectura es un ORF

            Information inicioATG = lectura.getStart();            
            System.out.println("inicioATG:" + inicioATG.position);
            List<Information> regionExplorar = constructor.getInnerInfo(0, inicioATG.position);
            String regionAdnUTR5p = regionExplorar.toString();
            regionAdnUTR5p = regionAdnUTR5p.replaceAll("\\[", "").replaceAll(",", "").replaceAll("\\]", "").replaceAll(" ", "");

            // Se prepara la region que contiene a los posibles UTR5'
            regionAdnUTR5p = regionAdnUTR5p.toUpperCase();
            File fileRegionUTR5p = new File("regionUTR5p.txt");
            fileRegionUTR5p.delete();
            metaData.guardar(regionAdnUTR5p, fileRegionUTR5p);
            
            Region regionPromotora = new Region(regionAdnUTR5p);

            ArrayList<ArrayList<Motivo>> corePromoters;
            
            if (consensos) { // Si consensos = true, se construye region promotora por consensos.
                regionPromotora.constructPromotorConsensos(regionAdnUTR5p, true);//  False construye promotor para region UTR3p.

            } else { // Si cnsensos es false se construye mediante FT obtenidos desde BioPattern.
                BioPattern pipeline = new BioPattern(regionAdnUTR5p, regionAdnUTR5p);
                regionPromotora = pipeline.pipelineBioPatternRP("regionUTR5p.txt", "0.90", 0, 0);

            }

            corePromoters = definirCisCorePromoters(regionPromotora, consensos, metaData);

            // Se definen posibles motivos que contienen factores que reconocen compsPromotsUniHUGO_ID Iniciador (Inr) o DPE.
            String[] factoresInrDPECorePromoter = {"TFIID", "TFIII", "TBP", "TAF", "USF", "E2F", "B-TFIID"};

            ArrayList<Motivo> posiblesMotivosInrDPE = new ArrayList<>();

            for (Motivo m : regionPromotora.getPromotor()) {

                for (factorTranscripcion ft : m.getFactores()) {

                    for (String fInr : factoresInrDPECorePromoter) {
                        if (ft.getID().indexOf(fInr) != -1) {
                            posiblesMotivosInrDPE.add(m);
                            break;
                        }
                    }

                }

            }

            //int distanciaMinInr_DPE = 28;
            //int distanciaMaxInr_DPE = 32;
            //int sizeCorePromoter;
            int coordTSS;
            int minUTR5p;

            if (!corePromoters.isEmpty()) {

                for (ArrayList<Motivo> corePromoter : corePromoters) {// Que pasa si no hay core promoter?:

                    coordTSS = 0;

                    if (inrILP) { // Se invoca al predictor ILP para que proponga TSS.
                        utr5pDefined = false;
                        /*int coordTSSp1EPD = definirCoordTSS(corePromoter, lectura);// Se invoca al predictor para que proponga TSS.
                         // Se asigna UTRs5p al core promoter en curso para lectura.
                         if ((coordTSSp1EPD != -1) && !coordsTSS.contains(coordTSSp1EPD)) {
                         utr5pDefined = asignarILPUTRs5p(lectura, coordTSSp1EPD, inicioUTR5p, genInformation, corePromoter);
                         coordsTSS.add(coordTSSp1EPD);
                         }*/
                        //coordsTSS.add(coordTTS);
                    } else { // La coordenada TSS se define por consenso Inr, distancia DPE o modelo clasificador.

                        Motivo motivoInrDPE=null;
                        Motivo motivoDPE = null;

                        Motivo motivoSup = corePromoter.get(corePromoter.size() - 1);
                        String core = motivoSup.getCore();

                        if (motivoSup.getCore().equals("Inr-tata-like")) {

                            motivoInrDPE = motivoSup;

                            coordTSS = motivoInrDPE.getCoordenadas()[0] + 2;
                            corePromoter.add(motivoInrDPE);

                            motivoDPE = definirCoordInrDPETSS(corePromoter, posiblesMotivosInrDPE, false);

                            if (motivoDPE != null) {
                                motivoDPE.setCore("DPE");
                                corePromoter.add(motivoDPE);
                                System.out.println("El Gen " + metaData.get_GenID().get(0) + " posee caja DPE en coordenada: " + motivoDPE.getCoordenadas()[0]);
                            }

                            // Se asigna UTR5' al core promoter en curso para lectura u ORF en curso.
                            minUTR5p = inicioATG.position - coordTSS;

                            if ((minUTR5p >= Model.minUTR5p) && (minUTR5p <= Model.maxUTR5p)) {
                                utr5pDefined = asignarILPUTRs5p(lectura, core, coordTSS, inicioATG, genInformation, corePromoter);
                                coordsTSS.add(coordTSS);
                            }

                        } else if (motivoSup.getCore().equals("DPE-tata-like")) {

                            coordTSS = definirTSS(motivoSup, inicioATG.position, constructor);

                            System.out.println("El Gen " + metaData.get_GenID().get(0) + " posee caja DPE en coordenada: " + motivoDPE.getCoordenadas()[0]);

                            corePromoter.add(motivoSup);

                            minUTR5p = inicioATG.position - coordTSS;

                            if ((minUTR5p >= Model.minUTR5p) && (minUTR5p <= Model.maxUTR5p)) {
                                utr5pDefined = asignarILPUTRs5p(lectura, core, coordTSS, inicioATG, genInformation, corePromoter);
                                coordsTSS.add(coordTSS);
                            }

                        }

                        if (!posiblesMotivosInrDPE.isEmpty() && (coordTSS == 0)) {

                            motivoInrDPE = definirCoordInrDPETSS(corePromoter, posiblesMotivosInrDPE, true);                            

                            if (motivoInrDPE != null) {

                                coordTSS = motivoInrDPE.getCoordenadas()[0] + 2;

                                core = motivoInrDPE.getCore();

                                minUTR5p = inicioATG.position - coordTSS;

                                if ((minUTR5p >= Model.minUTR5p) && (minUTR5p <= Model.maxUTR5p)) {

                                    corePromoter.add(motivoInrDPE);
                                    
                                    System.out.println("El Gen " + metaData.get_GenID().get(0) + " posee caja Inr en coordenada: " + coordTSS);

                                    // Se define motivo DPE para el core promoter en curso.
                                    motivoDPE = definirCoordInrDPETSS(corePromoter, posiblesMotivosInrDPE, false);

                                    if (motivoDPE != null) {
                                        motivoDPE.setCore("DPE");
                                        corePromoter.add(motivoDPE);
                                        System.out.println("El Gen " + metaData.get_GenID().get(0) + " posee caja DPE en coordenada: " + motivoDPE.getCoordenadas()[0]);
                                    }

                                    // Se asigna UTRs5p al core promoter en curso para lectura.                                  
                                    utr5pDefined = asignarILPUTRs5p(lectura, core, coordTSS, inicioATG, genInformation, corePromoter);
                                    coordsTSS.add(coordTSS);
                                }

                            } else {

                                motivoDPE = definirCoordInrDPETSS(corePromoter, posiblesMotivosInrDPE, false);
                                
                                if (motivoDPE != null) {

                                    coordTSS = definirTSS(motivoSup, motivoDPE.getCoordenadas()[0], constructor);

                                    minUTR5p = inicioATG.position - coordTSS;      // entre coord del DPE y la del ATG
                                    core = "DPE";
                                    if ((minUTR5p >= Model.minUTR5p) && (minUTR5p <= Model.maxUTR5p)) {
                                        motivoDPE.setCore("DPE");
                                        corePromoter.add(motivoDPE);
                                        System.out.println("El Gen " + metaData.get_GenID().get(0) + " posee caja DPE en coordenada: " + motivoDPE.getCoordenadas()[0]);

                                        utr5pDefined = asignarILPUTRs5p(lectura, core, coordTSS, inicioATG, genInformation, corePromoter);
                                        coordsTSS.add(coordTSS);
                                    }

                                } else if (coordTSS == 0) {

                                    /*
                                Dado que no hay Inr o DPE posibles, aqui se aplicaran modelos para asignar TSS 
                                delante del core promoter y antes del ATG
                                Para asignar coordTSS se empleara metodo que reciba el motivo mas a la derecha
                                del corepromoter y la coordenada del ATG y, entre ambas, se asignara el mejor TSS segun los TSS
                                ya disponibles desde el clasificador. El metodo recibira la coordenada de cierre en la exploracion
                                y si se debe elegir por ATG o DPE.                                
                                     */
                                    coordTSS = definirTSS(motivoSup, inicioATG.position, constructor);

                                    if (coordTSS != 0) {
                                        minUTR5p = inicioATG.position - coordTSS;
                                        if ((minUTR5p >= Model.minUTR5p) && (minUTR5p <= Model.maxUTR5p)) {
                                            utr5pDefined = asignarILPUTRs5p(lectura, core, coordTSS, inicioATG, genInformation, corePromoter);
                                            coordsTSS.add(coordTSS);
                                        }
                                    }

                                }
                            }

                        } else if (coordTSS == 0) {  // Dado que no hay Inr o DPE posibles, aqui se aplicaran modelos para asignar TSS 
                            // delante del core promoter y antes del ATG
                            /*
                                Para asignar coordTSS se empleara metodo que reciba el motivo mas a la derecha
                                del corepromoter y la coordenada del ATG y, entre ambas, se asignara el mejor TSS segun los TSS
                                ya disponibles desde el clasificador. El metodo recibira la coordenada de cierre en la exploracion
                                y si se debe elegir por ATG o DPE.        
                             */
                            coordTSS = definirTSS(motivoSup, inicioATG.position, constructor);

                            if (coordTSS != 0) {
                                minUTR5p = inicioATG.position - coordTSS;
                                if ((minUTR5p >= Model.minUTR5p) && (minUTR5p <= Model.maxUTR5p)) {
                                    utr5pDefined = asignarILPUTRs5p(lectura, core, coordTSS, inicioATG, genInformation, corePromoter);
                                    coordsTSS.add(coordTSS);
                                }
                            }
                        }
                    }
                }
            } else { // Caso en el que no hay promotor pero se asigna TSS mediante los modelos del clasificador.
                     // Debe revisarse el codigo del metodo signarILPUTRs5 para este caso especial. 

                coordTSS = definirTSS(null, inicioATG.position, constructor);

                minUTR5p = inicioATG.position - coordTSS;

                if (coordTSS != 0) {

                    if ((minUTR5p >= Model.minUTR5p) && (minUTR5p <= Model.maxUTR5p)) {
                        String core = "no-promoter";
                        utr5pDefined = asignarILPUTRs5p(lectura, core, coordTSS, inicioATG, genInformation, null);
                        coordsTSS.add(coordTSS);
                    }
                }

            }

        }
        return utr5pDefined;

    }

    /**
     * Metodo usado para la construccion de Regiones 5' para transcrito en
     * proceso. Ya se tiene un ORF y se desea completar el transcrito
     * correspondiente.
     */
    public Region constructORFListAbts(Utilities metaData, GenInformation genInformation, Gene utr5p, Gene gene, String estructura, String pathEstructura, int numObjs, int numIter) throws Exception {

        //String fileAbID = estructura + ".abst";
        System.out.println("Definiendo listado de FTs para: " + estructura);

        Information inicioUTR5p = utr5p.getStart();
        int iniRegionPromo, finRegionProm;

        //System.out.println("inicioUTR5p:" + inicioUTR5p.position);
        iniRegionPromo = inicioUTR5p.position - Model.limInfRegionPromo;
        finRegionProm = inicioUTR5p.position + Model.limSupRegionPromo;
        Information coordIniRegPromo;
        Information coordFinRegPromo;

        if (iniRegionPromo > 0) {
            coordIniRegPromo = constructor.getData(iniRegionPromo);
        } else {
            coordIniRegPromo = constructor.getData(0);
        }

        Information coordATG = gene.getStart();

        if (finRegionProm > coordATG.position) {
            coordFinRegPromo = coordATG;
        } else {
            coordFinRegPromo = constructor.getData(finRegionProm);
        }

        List<Information> regionExplorar = constructor.getInnerInfo(coordIniRegPromo.position, coordFinRegPromo.position);
        String regionPromotoraORF = regionExplorar.toString();
        regionPromotoraORF = regionPromotoraORF.replaceAll("\\[", "").replaceAll(",", "").replaceAll("\\]", "").replaceAll(" ", "");

        // Se preparan los datos a pasar al pipeline
        regionPromotoraORF = regionPromotoraORF.toUpperCase();
        
        String regionUTR5p = pathEstructura + "/" + estructura + ".rg";
        File fileRegionUTR5p = new File(regionUTR5p);
        fileRegionUTR5p.delete();
        metaData.guardar(regionPromotoraORF, fileRegionUTR5p);

        // Se invoca al pipeline quien devuelve la region con el promotor propuesto para la secuencia problema.
        // El promotor es una coleccion de motivos, cada uno con la coleccion de factores de transcripcion
        // que le reconocen. Alli deben estar los factores que definen un core promoter.
        BioPattern pipeline = new BioPattern(regionUTR5p, regionUTR5p);
        //Region regionPromotora = pipeline.pipelineBioPattern("regionProm.txt", "regionProm.txt", "0.95", 10, numObjs, numIter, estructura + ".txt", true);
        Region regionPromotora = pipeline.pipelineBioPatternRP(regionUTR5p, "0.95", 0, 0);

        String transFactFileID = pathEstructura + "/" + estructura + ".tf";

        File transFTfileID = new File(transFactFileID);

        String listsFacts = pathEstructura + "/" + estructura + ".pl";

        File listFts = new File(listsFacts);
        listFts.delete();

        System.out.println("Listado de factores de transcripción para:" + estructura);
        metaData.guardar("Listado de factores de transcripción para:" + estructura + "\n", transFTfileID);
        System.out.println("Motivo:\t\t\t\tCoords:\t\tFactor Simbolo:\tFactor Name:");
        metaData.guardar("Motivo:\t\t\t\tCoords:\t\tFactor Simbolo:\tFactor Name:\n", transFTfileID);

        ArrayList<String> listFactores = new ArrayList<>();

        String factorID, factorName, factorSimb;

        int[] coordnsMotivo;

        for (Motivo motivo : regionPromotora.getPromotor()) {

            coordnsMotivo = motivo.getCoordenadas();

            coordnsMotivo[0] = coordnsMotivo[0] + coordIniRegPromo.position;
            coordnsMotivo[1] = coordnsMotivo[1] + coordIniRegPromo.position;

            motivo.setCoordenadas(coordnsMotivo);

        }

        for (Motivo motivo : regionPromotora.getPromotor()) {

            String consenso = motivo.getFactores().get(0).getLecturasTFBIND().getCadena();

            int[] coordsMotivo = motivo.getCoordenadas();

            ArrayList<factorTranscripcion> fts = motivo.getFactores();

            factorID = fts.get(0).getID();
            factorName = fts.get(0).getID();
            factorSimb = fts.get(0).getID();

            System.out.println(consenso + "\t\t" + coordsMotivo[0] + "-" + coordsMotivo[1] + "\t\t" + factorSimb + "\t" + factorName);
            metaData.guardar(consenso + "\t\t" + coordsMotivo[0] + "-" + coordsMotivo[1] + "\t\t" + factorSimb + "\t" + factorName, transFTfileID);

            if (!listFactores.contains(factorID)) {
                listFactores.add(factorID);
            }
            if (!listFactores.contains(factorSimb)) {
                listFactores.add(factorSimb);
            }
            if (!listFactores.contains(factorName)) {
                listFactores.add(factorName);
            }

        }

        int cantObsFat = listFactores.size(), contIDs = 0;
        String listaFTstrings = "[";

        for (String s : listFactores) {
            if (contIDs != (cantObsFat - 1)) {
                listaFTstrings = listaFTstrings + "'" + s + "',";
                contIDs++;
            } else {
                listaFTstrings = "listFTs(" + listaFTstrings + "'" + s + "']).";
            }
        }

        System.out.println("Lista de factores de transcripción para:" + estructura);
        System.out.println(listaFTstrings);
        metaData.guardar(listaFTstrings, listFts);

        return regionPromotora;

    }

    /**
     * Metodo usado para la construccion de Regiones 5' para transcrito en
     * proceso. Ya se tiene un ORF y se desea completar el transcrito
     * correspondiente.
     */
    public void listUTR5pHeader(Utilities metaData, Gene gene, Gene utr5p, int coorATG, String fileAbstID, String tipoReporte, File transFTfileID) throws Exception {

        //String transFactFileID = fileAbstID + ".tf";
        metaData.guardar("Coordenadas region reguladora para: " + fileAbstID + "\n", transFTfileID);
        metaData.guardar("Coord ATG:\tCoordTSSp1:\tTipoCoreProm:\tCoorIniRegReg:\tCoordFinRegReg:", transFTfileID);

        int[] coordsRegionReg = {0, 0};
        int coordTSSp1;

        if (tipoReporte.equals("5p3p") || tipoReporte.equals("5p")) {
            List<Motivo> corePromoter = utr5p.getPromoter();

            coordTSSp1 = utr5p.getStart().position;

            String tipoCoreProm = "";

            for (Motivo m : corePromoter) {
                tipoCoreProm = tipoCoreProm + m.getCore() + " ";
            }

            coordsRegionReg[0] = coordTSSp1 - Model.limInfRegionPromo;
            coordsRegionReg[1] = coordTSSp1 + Model.limSupRegionPromo;

            if (coordsRegionReg[0] < 0) {
                coordsRegionReg[0] = 0;
            }

            int coordATG = gene.getStart().position;

            if (coordsRegionReg[1] > coordATG) {
                coordsRegionReg[1] = coordATG - 1;
            }

            metaData.guardar(coorATG + "\t\t" + coordTSSp1 + "\t\t" + tipoCoreProm + "\t" + coordsRegionReg[0] + "\t" + coordsRegionReg[1] + "\n", transFTfileID);

            metaData.guardar("Motivos core promoter: " + "\n", transFTfileID);

            String motifFirma, consenso;

            for (Motivo m : corePromoter) {
                metaData.guardar(m.getCore() + ": ", transFTfileID);
                motifFirma = m.getFactores().get(0).getLecturasTFBIND().getCadena();
                consenso = m.getFactores().get(0).getLecturasTFBIND().getCadena();
                metaData.guardar("Motif: " + motifFirma, transFTfileID);
                metaData.guardar("Coords: " + m.getCoordenadas()[0] + "-" + m.getCoordenadas()[1] + "\n", transFTfileID);
            }

        } else {
            coordTSSp1 = coorATG;
            coordsRegionReg[0] = coordTSSp1 - Model.limInfRegionPromo;
            if (coordsRegionReg[0] < 0) {
                coordsRegionReg[0] = 0;
            }
            coordsRegionReg[1] = coordTSSp1 - 1;
            metaData.guardar(coorATG + "\t" + coorATG + "\t" + "NA" + "\t" + coordsRegionReg[0] + "\t" + coordsRegionReg[1] + "\n", transFTfileID);

        }

    }

    /**
     * Metodo usado para la construccion de Regiones 5' para transcrito en
     * proceso. Ya se tiene un ORF y se desea completar el transcrito
     * correspondiente.
     */
    public void constructUTR5pProximalPromoters(Utilities metaData, GenInformation genInformation) throws Exception {

        String genID = metaData.get_GenID().get(0);

        int contORFs = 0, contUTR5ps = 0;
        String fileAbstID;

        for (Gene lectura : lectures) {

            List<Gene> utrp5ps = lectura.getUtr5ps();

            for (Gene utr5p : utrp5ps) {

                fileAbstID = genID + "-abts-ORF-UTR5p-" + contORFs + "-UTR5p-" + contUTR5ps;

                System.out.println("Definiendo listado de abstracts para UTR5p: " + fileAbstID);

                Information inicioUTR5p = utr5p.getStart();
                System.out.println("inicioUTR5p:" + inicioUTR5p.position);
                List<Information> regionExplorar = constructor.getInnerInfo(0, inicioUTR5p.position);
                String regionProximalPromoterUTR5p = regionExplorar.toString();
                regionProximalPromoterUTR5p = regionProximalPromoterUTR5p.replaceAll("\\[", "").replaceAll(",", "").replaceAll("\\]", "").replaceAll(" ", "");

                // Se preparan los datos a pasar al pipeline
                regionProximalPromoterUTR5p = regionProximalPromoterUTR5p.toUpperCase();
                File fileRegionUTR5p = new File("regionProxPromUTR5p.txt");
                fileRegionUTR5p.delete();
                metaData.guardar(regionProximalPromoterUTR5p, fileRegionUTR5p);

                // Se invoca al pipeline quien devuelve la region con el promotor propuesto para la secuencia problema.
                // El promotor es una coleccion de motivos, cada uno con la coleccion de factores de transcripcion
                // que le reconocen. Alli deben estar los factores que definen un core promoter.
                BioPattern pipeline = new BioPattern(regionProximalPromoterUTR5p, regionProximalPromoterUTR5p);
                Region regionPromotora = pipeline.pipelineBioPatternRP("regionProxPromUTR5p.txt", "0.85", 0, 0);

            }
        }

    }

    /**
     * Metodo usado para la construccion de Regiones 3' para transcrito en
     * proceso. Ya se tiene un ORF y se desea completar el transcrito
     * correspondiente.
     */
    public boolean constructRegionUTR3p(Utilities metaData, GenInformation genInformation, boolean ilpConsensos) throws Exception {

        boolean utr3pDefined = false;
        ArrayList<Integer> coordsTTS = new ArrayList<>(); // Posibles Transcription Termination Sites.

        for (Gene lectura : lectures) {
            Information stop = lectura.getEnd();
            System.out.println("Stop:" + stop.position);
            List<Information> regionExplorar = constructor.getInnerInfo(stop.position + 1, constructor.lastData() + 1);

            String regionAdnUTR3p = regionExplorar.toString();
            regionAdnUTR3p = regionAdnUTR3p.replaceAll("\\[", "").replaceAll(",", "").replaceAll("\\]", "").replaceAll(" ", "");

            // Se preparan los datos a pasar al pipeline
            regionAdnUTR3p = regionAdnUTR3p.toUpperCase();
            File fileRegionUTR3p = new File("regionUTR3p.txt");
            fileRegionUTR3p.delete();
            metaData.guardar(regionAdnUTR3p, fileRegionUTR3p);

            // Se invoca al pipeline quien devuelve la region con el promotor propuesto para la secuencia problema.
            // El promotor es una coleccion de motivos, cada uno con la coleccion de factores de transcripcion
            // que le reconocen. Alli deben estar los factores que definen un core promoter.
            Region regionPromotora = new Region(regionAdnUTR3p);

            regionPromotora.constructPromotorConsensos(regionAdnUTR3p, false); // Falta agregar que se detecten cajas DSE      

            // En el array corePromoters estan los promotores que estan del lado izquierdo de la caja TATA y que la 
            // incluyen (cis promoters).
            ArrayList<ArrayList<Motivo>> uTR3pCorePromoters = definirUTR3p_CorePromoters(regionPromotora, true, metaData);

            ArrayList<ArrayList<Motivo>> corePromotersDSE = new ArrayList<>();

            // Se definen posibles motivos que contienen factores que reconocen compsPromotsUniHUGO_ID Iniciador (Inr) o DPE.
            String[] factoresDSECorePromoter = {"CSTF", "Cstf", "cleavage stimulation factor", "Q05048", "A3KFI9", "Q5QPD8"};

            ArrayList<Motivo> posiblesMotivosDSE = new ArrayList<>();

            for (Motivo m : regionPromotora.getPromotor()) {

                for (factorTranscripcion ft : m.getFactores()) {

                    for (String fDSE : factoresDSECorePromoter) {
                        if (ft.getID().indexOf(fDSE) != -1) {
                            posiblesMotivosDSE.add(m);
                            break;
                        }
                    }

                }

            }

            int coordTTS = 0;
            int minUTR3p = 0;

            if (!uTR3pCorePromoters.isEmpty()) {

                for (ArrayList<Motivo> uTR3CorePromoter : uTR3pCorePromoters) {

                    Motivo motivoSup = uTR3CorePromoter.get(uTR3CorePromoter.size() - 1);
                    String core = motivoSup.getCore();

                    if (ilpConsensos) { // Se invoca al predictor para que proponga TTS.
                        utr3pDefined = false;
                        /*
                         int coordTTS = definirCoordTTS(uTR3CorePromoter, lectura);// Se invoca al predictor para que proponga TSS.
                         // Se asigna UTRs3p a la lectura en curso.
                         if ((coordTTS != -1) && !coordsTTS.contains(coordTTS)) {
                         utr3pDefined = asignarILPUTRs3p(lectura, coordTTS, stop, uTR3CorePromoter);
                         coordsTTS.add(coordTTS);
                         }
                         //coordsTSS.add(coordTTS);*/
                    } else // La coordenada TTS se define por transicion GC o distancia estadistica.
                     if (!posiblesMotivosDSE.isEmpty()) {

                            Motivo motivoDSE = definirMotivoDSE(uTR3CorePromoter, posiblesMotivosDSE);

                            if (motivoDSE != null) {

                                coordTTS = this.definirTTSClasificador(motivoSup, motivoDSE, stop.position, constructor, lectura);

                                if (coordTTS != 0) {
                                    uTR3CorePromoter.add(motivoDSE);
                                    //corePromotersDSE.add(uTR3CorePromoter);

                                    // Se asigna UTRs5p al core promoter en curso para lectura.
                                    //if (!coordsTTS.contains(coordTTS)) {
                                    utr3pDefined = asignarILPUTRs3p(lectura, coordTTS, stop, uTR3CorePromoter);
                                    coordsTTS.add(coordTTS);
                                    //}
                                }
                            } else {

                                coordTTS = this.definirTTSClasificador(motivoSup, null, stop.position, constructor, lectura);

                                if (coordTTS != 0) {

                                    corePromotersDSE.add(uTR3CorePromoter);

                                    // Se asigna UTRs5p al core promoter en curso para lectura.
                                    //if (!coordsTTS.contains(coordTTS)) {
                                    utr3pDefined = asignarILPUTRs3p(lectura, coordTTS, stop, uTR3CorePromoter);
                                    coordsTTS.add(coordTTS);
                                    //}
                                }
                            }
                        } else {

                            coordTTS = this.definirTTSClasificador(motivoSup, null, stop.position, constructor, lectura);

                            if (coordTTS != 0) {

                                corePromotersDSE.add(uTR3CorePromoter);

                                // Se asigna UTRs5p al core promoter en curso para lectura.
                                //if (!coordsTTS.contains(coordTTS)) {
                                utr3pDefined = asignarILPUTRs3p(lectura, coordTTS, stop, uTR3CorePromoter);
                                coordsTTS.add(coordTTS);
                                //}
                            }
                        }
                }
            } else {

                // Caso en el que no hay promotor pero se asigna TSS mediante los modelos del clasificador.
                // Debe revisarse el codigo del metodo signarILPUTRs5 para este caso especial.
                coordTTS = this.definirTTSClasificador(null, null, stop.position, constructor, lectura);

                if (coordTTS != 0) {

                    // Se asigna UTRs5p al core promoter en curso para lectura.
                    //if (!coordsTTS.contains(coordTTS)) {
                    utr3pDefined = asignarILPUTRs3p(lectura, coordTTS, stop, null);
                    coordsTTS.add(coordTTS);
                    //}
                }

            }
        }

        return utr3pDefined;
    }

    /**
     * Metodo usado para la construccion de Regiones 3' para transcrito en
     * proceso. Ya se tiene un ORF y se desea completar el transcrito
     * correspondiente.
     */
    public boolean asignarILPUTRs5p(Gene lectura, String core, int coordTSS, Information inicioATG, GenInformation genInformation, ArrayList<Motivo> corePromoter) throws IOException, Exception {

        boolean utrs5p = false;

        //List<Information> regionUTR5p = constructor.getInnerInfo(coordTSSp1EPD + 1, inicioUTR5p.position);
        /*Analizer analizadorUTR5p = genInformation.generaLectsUTR5p(regionUTR5p);
         // !analizadorUTR5p.getLectures().isEmpty()
         if (false) {// Ojo: Se pueden hacer vacios los ATGs y Stops y volver a predecir.
         for (Gene utr : analizadorUTR5p.getLectures()) {
         utr.setPromotor(corePromoter);
         lectura.getUtr5ps().add(utr);

         }
         utrs5p = true;

         } else {*/
 /*
         Information inicioGen = constructor.getGeneData().get(1075);
         Information finGen = constructor.getGeneData().get(1235);
         //List<Information> genData = analizer.getLectures().get(0).getData();        
         List<Information> genData = constructor.getInnerInfo(0, constructor.lastData() - 1);
         Gene lectura = new Gene(inicioGen, finGen, true, genData);
         //String temp = lectura.toString();
         Information inicioExon = constructor.getData(1075);
         Information finExon = constructor.getData(1237);
         List<Information> innnerExon = constructor.getInnerInfo(inicioExon.position + 1, finExon.position);
         Exon exon = new Exon(inicioExon, finExon, innnerExon);
         lectura.addExon(exon);*/
        Information inicioPlus1 = constructor.getData(coordTSS + 1);
        Information finUTR5p = constructor.getData(inicioATG.position - 1);
        List<Information> innerInfoUTR5p = constructor.getInnerInfo(inicioPlus1.position + 1, finUTR5p.position);
        Gene utr5p = new Gene(inicioPlus1, finUTR5p, innerInfoUTR5p);

        Information inicioExon = constructor.getData(coordTSS + 1);
        Information finExon = constructor.getData(inicioATG.position - 1);
        List<Information> innnerExon = constructor.getInnerInfo(inicioExon.position + 1, finExon.position);
        Exon exon = new Exon(inicioExon, finExon, innnerExon);

        utr5p.setCore(core);

        utr5p.addExon(exon);

        utr5p.setPromotor(corePromoter);
        lectura.getUtr5ps().add(utr5p);
        utrs5p = true;
        //}
        // Definida esta region UTR5p se procede a definir el proximal promoter correspondiente.

        return utrs5p;
    }

    /**
     * Metodo usado para la construccion de Regiones 3' para transcrito en
     * proceso. Ya se tiene un ORF y se desea completar el transcrito
     * correspondiente.
     */
    public boolean asignarILPUTRs3p(Gene lectura, int coordTTS, Information stop, ArrayList<Motivo> motifsUTR3p) throws IOException, Exception {

        boolean utrs3p;
        try {
            int inicioUTR3p = stop.position + 1;
            int finutr3p = coordTTS - 1;
            Information stopPlus3 = constructor.getData(inicioUTR3p);
            Information finUTR3p = constructor.getData(finutr3p);
            List<Information> innerInfoUTR3p = constructor.getInnerInfo(stopPlus3.position + 1, finUTR3p.position + 3);
            UTR3p utr3p = new UTR3p(stopPlus3, finUTR3p, innerInfoUTR3p);
            utr3p.setPromoter(motifsUTR3p);
            lectura.getUtr3p().add(utr3p);
            utrs3p = true;
        } catch (Exception e) {
            e.printStackTrace();
            utrs3p = false;

        }
        return utrs3p;
    }

    /**
     * Metodo usado para la construccion de Regiones 5' para transcrito en
     * proceso. Ya se tiene un ORF y se desea completar el transcrito
     * correspondiente.
     */
    public int definirCoordTSS(ArrayList<Motivo> corePromoter, Gene lectura) throws Exception {

        int coordSupMotSup, coordGobalTSS = -1;

        if (!corePromoter.isEmpty()) {

            Motivo motivoSup = corePromoter.get(corePromoter.size() - 1);
            coordSupMotSup = motivoSup.getCoordenadas()[1];

        } else {

            coordSupMotSup = -1;

        }

        List<Information> regionTSS = constructor.getInnerInfo(coordSupMotSup + 1, lectura.getStart().position);

        ArrayList<String> rTSS = new ArrayList<>();
        for (Information i : regionTSS) {
            rTSS.add(i.toString());
        }

        String rTSStemp = rTSS.toString().replaceAll(" ", "");

        String tssRegion = "region_tss(" + rTSStemp + ").";

        File fileRegionUTR5p = new File("regionTSS.pl");
        fileRegionUTR5p.delete();
        try (BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(fileRegionUTR5p, true), "UTF8"))) {
            out.write(tssRegion);
            out.write("\n");
            out.close();
        }

        MiddleWare middle = new MiddleWare();
        middle.init("p_genes.pl");

        String rTSConsensos = rTSStemp.replaceAll("\\[", "").replaceAll(",", "").replaceAll("\\]", "");
        //List<Integer> coordsTSS = middle.consultTSSs();// Atento que al hace la prediccion TSS el vector o sera' absoluto.
        List<Integer> coordsTSS = middle.consultTSSsInr(rTSConsensos);

        if (coordsTSS.isEmpty()) {

            coordsTSS = middle.consultTSSsDPE(rTSConsensos);
        }

        int distancia_TATA_TSS, coord_TSS = -1;

        int umbral_region_tss_inf = 25; // Segun conocimiento experto el transcription start site esta' a esa distancia umbral
        int umbral_region_tss_sup = 200; // Segun conocimiento experto el transcription start site esta' a esa distancia umbral

        if (coordSupMotSup != -1 && (!coordsTSS.isEmpty())) {
            for (Integer coord_tss : coordsTSS) {

                distancia_TATA_TSS = coord_tss + 1;
                if ((distancia_TATA_TSS > 0) && (distancia_TATA_TSS < umbral_region_tss_sup) && (distancia_TATA_TSS > umbral_region_tss_inf)) {
                    coord_TSS = coord_tss;
                    break; // Tomamos el TSS mas cercano al core promoter.
                }

            }

            if (coord_TSS != -1) {
                coordGobalTSS = coord_TSS + coordSupMotSup;
            }

        } else if (coordSupMotSup == -1 && (!coordsTSS.isEmpty())) {
            coordGobalTSS = coordsTSS.get(0);
        }

        return coordGobalTSS;
    }

    /**
     * Metodo usado para la construccion de Regiones 5' para transcrito en
     * proceso. Ya se tiene un ORF y se desea completar el transcrito
     * correspondiente.
     */
    public int definirCoordTTS(ArrayList<Motivo> utr3pMotifs, Gene lectura) throws Exception {

        int coordSupMotSup, coordTTSGlobal = -1;
        Motivo motivoSup;

        if (!utr3pMotifs.isEmpty()) {
            int utr3pSize = utr3pMotifs.size();
            if (utr3pSize == 3) {

                motivoSup = utr3pMotifs.get(utr3pMotifs.size() - 2);

            } else {

                motivoSup = utr3pMotifs.get(utr3pMotifs.size() - 1);
            }
            coordSupMotSup = motivoSup.getCoordenadas()[1];

        } else {

            coordSupMotSup = -1;

        }
        int stopRef = lectura.getEnd().position + 3;
        List<Information> regionTTS = constructor.getInnerInfo(stopRef + coordSupMotSup + 1, constructor.lastData() + 1);

        ArrayList<String> rTTS = new ArrayList<>();
        for (Information i : regionTTS) {
            rTTS.add(i.toString());
        }

        String rTTStemp = rTTS.toString().replaceAll(" ", "");

        String ttsRegion = "region_tts(" + rTTStemp + ").";

        File fileRegionUTR3p = new File("regionTTS.pl");
        fileRegionUTR3p.delete();
        try (BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(fileRegionUTR3p, true), "UTF8"))) {
            out.write(ttsRegion);
            out.write("\n");
            out.close();
        }

        MiddleWare middle = new MiddleWare();
        middle.init("p_genes.pl");

        //List<Integer> coordsTTS = middle.consultTTSs();
        List<Integer> coordsTTS = middle.consultTTSsPolyA(rTTStemp);

        if (coordsTTS.isEmpty()) {
            coordsTTS = middle.consultTTSsDSE(rTTStemp);
        }

        int distancia_PoliAdenil_TTS, coord_TTS = -1;

        int umbral_region_tts_inf = 10; // Segun conocimiento experto el transcription start site esta' a esa distancia umbral
        int umbral_region_tts_sup = 200; // Segun conocimiento experto el transcription start site esta' a esa distancia umbral

        if (coordSupMotSup != -1 && (!coordsTTS.isEmpty())) {
            for (Integer coord_tts : coordsTTS) {

                distancia_PoliAdenil_TTS = coord_tts;
                if ((distancia_PoliAdenil_TTS > 0) && (distancia_PoliAdenil_TTS < umbral_region_tts_sup) && (distancia_PoliAdenil_TTS > umbral_region_tts_inf)) {
                    coord_TTS = coord_tts;
                    break; // Tomamos el TSS mas cercano al core promoter.
                }
            }

            if (coord_TTS != -1) {
                coordTTSGlobal = coord_TTS + stopRef + coordSupMotSup;
            }

        } else if (coordSupMotSup == -1 && (!coordsTTS.isEmpty())) {
            coordTTSGlobal = coordsTTS.get(0) + stopRef;
        }

        return coordTTSGlobal;

    }

    /**
     * Metodo usado para la construccion de Regiones 5' para transcrito en
     * proceso. Ya se tiene un ORF y se desea completar el transcrito
     * correspondiente.
     */
    private boolean hasConsenseInrDPE(boolean inr, Motivo posibleInrDPE) {

        boolean hasConsense = false;

        String motif = posibleInrDPE.getMotivo();
        //String motifFirma = "AGTATAAAAG";
        if (inr) {
            hasConsense = Pattern.matches("[TG]C[AT][GTC][TCA][CT][TCG][TC]", motif);// Consenso Inr
            if (hasConsense) {
                posibleInrDPE.setCore("Inr");
            } else {
                String ftName = posibleInrDPE.getFactores().get(0).getID();
                if (ftName.indexOf("E2F") != -1) {
                    posibleInrDPE.setCore("E2F-Inr-Like");
                    hasConsense = true;
                }
                if (ftName.indexOf("USF") != -1) {
                    posibleInrDPE.setCore("USF-Inr-Like");
                    hasConsense = true;
                }
                String tataLike = posibleInrDPE.getCore();
                if ((ftName.indexOf("TAF") != -1) && (!tataLike.equals("TATA"))) {
                    posibleInrDPE.setCore("TAF-Inr-Like");
                    hasConsense = true;
                }
            }

            //hasConsense = Pattern.matches("[TA][GC][TA]A[ACGT][TA][A][A][A][GTC]", motifFirma);// Consenso de prueba
        } else { // Si no el consenso a evaluar es DPE.
            //hasConsense = Pattern.matches("[TA][GC][TA]A[ACGT][TA][A][A][A][GTC]", motifFirma);// Consenso de prueba
            hasConsense = Pattern.matches("[AG]G[AT][CT][GAC]", motif);// Consenso DPE
            posibleInrDPE.setCore("DPE");

        }

        return hasConsense;
    }

    /**
     * Metodo usado para la construccion de Regiones 5' para transcrito en
     * proceso. Ya se tiene un ORF y se desea completar el transcrito
     * correspondiente.
     */
    private boolean hasConsenseDSE(boolean dse, Motivo posibleDSE) {

        boolean hasConsense = false;

        String motif = posibleDSE.getMotivo();
        //String motifFirma = "AGTATAAAAG";

        if (dse) {
            //hasConsense = Pattern.matches("[TC][TC]A[ACGT][TA][TC][TC]", motifFirma);
            Pattern p = Pattern.compile("[CT]GTGTT[CT][CT]");
            Matcher m = p.matcher(motif);
            hasConsense = m.matches();
            //hasConsense = Pattern.matches("[GA]CACAA[GC][GC]", motifFirma);
        } else { // Si no el consenso a evaluar es DPE.
            //hasConsense = Pattern.matches("[AG]G[AT][CT][GAC]", motifFirma);
            //hasConsense = Pattern.matches("[AG]G[AT][CT][GAC]", "GGTCA");
            Pattern p = Pattern.compile("[AG]G[AT][CT][GAC]");
            Matcher m = p.matcher(motif);
            hasConsense = m.matches();

        }

        return hasConsense;
    }

    /**
     * Metodo usado para la construccion de Regiones 5' para transcrito en
     * proceso. Ya se tiene un ORF y se desea completar el transcrito
     * correspondiente.
     */
    public Motivo definirCoordInrDPETSS(ArrayList<Motivo> corePromoter, ArrayList<Motivo> posiblesMotivosInrDpe, boolean inrDPE) throws Exception {

        int coordSupCorePromoter;

        // Se define motivo del core promoter mas cercano al Inr o al DPE.
        Motivo motivoSup = corePromoter.get(corePromoter.size() - 1), motivoInrDPE = null;

        coordSupCorePromoter = motivoSup.getCoordenadas()[1];
        String core = motivoSup.getCore();
        int distMin = 0;

        if (inrDPE) {

            switch (core.toLowerCase()) {

                case "caat":
                    distMin = 120;
                    break;

                case "gc":
                    distMin = 120;
                    break;

                case "bre":
                    distMin = 25;
                    break;

                case "tata":
                    distMin = 20;
                    break;

                case "inr-tata-like":
                    distMin = 20;
                    break;

                case "dpe-tata-like":
                    distMin = 0;
                    break;

                default:
                    distMin = 0;
                    break;
            }
        } else {

            switch (core.toLowerCase()) {

                case "caat":
                    distMin = 140;
                    break;

                case "gc":
                    distMin = 140;
                    break;

                case "bre":
                    distMin = 45;
                    break;

                case "tata":
                    distMin = 40;
                    break;

                case "inr":
                    distMin = 20;
                    break;

                case "inr-tata-like":
                    distMin = 20;
                    break;

                case "dpe-tata-like":
                    distMin = 0;
                    break;

                default:
                    distMin = 0;
                    break;

            }
        }

        for (Motivo posibleInrDPE : posiblesMotivosInrDpe) {

            if ((posibleInrDPE.getCoordenadas()[0] > (coordSupCorePromoter + distMin)) && hasConsenseInrDPE(inrDPE, posibleInrDPE)) {
                motivoInrDPE = posibleInrDPE;
                break;
            }

        }

        return motivoInrDPE;

    }

    /**
     * Metodo usado para la construccion de Regiones 5' para transcrito en
     * proceso. Ya se tiene un ORF y se desea completar el transcrito
     * correspondiente.
     */
    public Motivo definirMotivoDSE(ArrayList<Motivo> corePromoter, ArrayList<Motivo> posiblesMotivosDSE) throws Exception {

        int coordSupCorePromoter, coordInfMotDSE, distanciaCorePromoterDSE;

        int distanciaMinCorePromoter_DSE = 10; // Distancia minima desde caja TATA a motivo BRE aguas arriba.
        int distanciaMaxCorePromoter_DSE = 200; // Distancia minima desde caja TATA a motivo BRE aguas arriba.

        // Se define motivo del core promoter mas cercano al Inr.
        Motivo motivoSup = corePromoter.get(corePromoter.size() - 1), motivoDSE = null;

        coordSupCorePromoter = motivoSup.getCoordenadas()[1];

        for (Motivo posibleDSE : posiblesMotivosDSE) {

            if ((posibleDSE.getCoordenadas()[0] > coordSupCorePromoter) && hasConsenseDSE(true, posibleDSE)) {
                coordInfMotDSE = posibleDSE.getCoordenadas()[0];
                distanciaCorePromoterDSE = coordInfMotDSE - coordSupCorePromoter;
                if ((distanciaMinCorePromoter_DSE < distanciaCorePromoterDSE) && (distanciaCorePromoterDSE < distanciaMaxCorePromoter_DSE)) {
                    motivoDSE = posibleDSE;
                    break;
                }
            }

        }

        return motivoDSE;

    }

    /**
     * Metodo usado para la construccion de Regiones 5' para transcrito en
     * proceso. Ya se tiene un ORF y se desea completar el transcrito
     * correspondiente.
     */
    public int definirTTSClasificador(Motivo motivoSupCorePromoter, Motivo motivoDSE, Integer stop, GeneConstructor constructor, Gene lectura) throws Exception {

        Integer TTS = 0, limSupExploracion = 0, limInfDSE = 60;
        int barrido = 0, distMin, distMax;
        boolean exploracion = true;
        double mayorProbTTS = 0.0;
        List<Integer> tts = constructor.getTts();
        ArrayList<Double> distPosTts = constructor.getDistPosTts();

        int motivoSupCoordenadas;

        if (motivoSupCorePromoter != null) {

            motivoSupCoordenadas = motivoSupCorePromoter.getCoordenadas()[1] + stop + 3;
            String core = motivoSupCorePromoter.getCore();

            switch (core.toLowerCase()) {

                // Distancias minimas y maximas desde el core promoter al TTS
                case "cpeb":
                    distMin = 130;
                    distMax = 140;
                    break;

                case "cpsf":
                    distMin = 110;
                    distMax = 120;
                    break;

                case "cstf":
                    distMin = 10;
                    distMax = 30;
                    break;

                default: // desde el stop al TTS
                    motivoSupCoordenadas = stop + 3;
                    distMin = Model.minUTR3p;
                    distMax = Model.maxUTR3p;
                    break;

            }

        } else {

            motivoSupCoordenadas = stop + 1;
            distMin = Model.minUTR3p;
            distMax = Model.maxUTR3p;
        }

        if (motivoDSE != null) {

            limSupExploracion = motivoDSE.getCoordenadas()[0] + stop + 1;

        } else if ((lectura.getEnd().position - stop + 1) > Model.maxUTR3p) {

            limSupExploracion = stop + 3 + Model.maxUTR3p;

        } else if ((lectura.getEnd().position - stop + 1) > Model.minUTR3p) {

            limSupExploracion = Model.minUTR3p + stop + 1;

        } else {

            limSupExploracion = lectura.getEnd().position + stop + 1;
        }

        for (Integer posicion : constructor.getTts()) {

            if ((posicion < (motivoSupCoordenadas + distMin))) {
                barrido++;
            } else {
                break;
            }

        }

        //if (barrido < tts.size() - 1) { 
        while (exploracion) {

            if (barrido < tts.size()) {

                if ((tts.get(barrido) <= (motivoSupCoordenadas + distMax)) && (tts.get(barrido) - stop + 3) >= Model.minUTR3p) {

                    if (motivoDSE != null) {

                        if (tts.get(barrido) < (motivoDSE.getCoordenadas()[0] - limInfDSE)) {

                            if (distPosTts.get(barrido) > mayorProbTTS) {
                                mayorProbTTS = distPosTts.get(barrido);
                                TTS = tts.get(barrido);
                            }
                        } else {
                            exploracion = false;
                        }
                    } else if (distPosTts.get(barrido) > mayorProbTTS) {
                        mayorProbTTS = distPosTts.get(barrido);
                        TTS = tts.get(barrido);
                    }

                    barrido++;

                } else {
                    exploracion = false;
                }

            } else {
                exploracion = false;
            }

        }

        return TTS;

    }

    /**
     * Metodo usado para la construccion de Regiones 5' para transcrito en
     * proceso. Ya se tiene un ORF y se desea completar el transcrito
     * correspondiente.
     */
    public int definirTTS(ArrayList<Motivo> corePromoter, Motivo motivoDSE, GeneConstructor constructor, Gene lectura) throws Exception {

        int coordSupCorePromoter, coordInfMotDSE, coordCA = -1;

        // Se define motivo del core promoter mas cercano al Inr.
        Motivo motivoSup = corePromoter.get(corePromoter.size() - 1);

        int stop = lectura.getEnd().position;

        coordSupCorePromoter = motivoSup.getCoordenadas()[1] + stop + 3;

        coordInfMotDSE = motivoDSE.getCoordenadas()[0] + stop + 3;

        List<Information> regionPosiblesCAs = constructor.getInnerInfo(coordSupCorePromoter + 1, coordInfMotDSE);
        String regionCAs = regionPosiblesCAs.toString();
        regionCAs = regionCAs.replaceAll(", ", "");
        int primerCA = regionCAs.indexOf("ca");
        int ultimoCA = regionCAs.lastIndexOf("ca");
        int indexCA = primerCA;
        ArrayList<Integer> posicionesCAs = new ArrayList<>();

        if (primerCA != -1) {

            do {
                posicionesCAs.add(indexCA);
                indexCA = regionCAs.indexOf("ca", indexCA + 2);
            } while (indexCA != ultimoCA);

            posicionesCAs.add(ultimoCA);

            int distanciaCorePromCA = 0;
            int distanciaCA_DSE = 0;

            for (Integer ca : posicionesCAs) {

                distanciaCorePromCA = ca;
                distanciaCA_DSE = coordInfMotDSE - (coordSupCorePromoter + ca);
                if ((10 <= distanciaCorePromCA) && (distanciaCorePromCA <= 30) && (distanciaCA_DSE <= 30)) {
                    coordCA = ca.intValue();
                    break;
                }

            }
        }

        if (coordCA != -1) {
            coordCA = coordCA + coordSupCorePromoter;

        } else {
            coordCA = 50 + coordSupCorePromoter; // La mas pobre de las posibilidades.
        }

        return coordCA;

    }

    /**
     * Metodo usado para la construccion de Regiones 5' para transcrito en
     * proceso. Ya se tiene un ORF y se desea completar el transcrito
     * correspondiente.
     */
    public ArrayList<ArrayList<Motivo>> definirCisCorePromoters(Region regionPromotora, boolean consenso, Utilities metaData) throws Exception {

        ArrayList<ArrayList<Motivo>> corePromoters = new ArrayList<>(); // Contedra' todos los core promoters propuestos.

        String[] factoresTATACorePromoter = {"TFIIA", "GTF2A1", "ALF", "TFIID", "TBP", "TAF1", "TAF2", "TATA", "B-TFIID", "GTF2D", "SCA17", "GTF"};
        String[] factoresCAATCorePromoter = {"NFY", "NF-Y", "C/EBP", "RNY", "HY", "CEBP", "CBF"};
        String[] factoresBRECorePromoter = {"TFIIB", "GTF2B"};
        String[] factoresGCCorePromoter = {"SP1", "Sp1 transcription factor", "specificity protein 1"};

        ArrayList<Motivo> motivosTATA = new ArrayList<>();
        ArrayList<Motivo> motivosBRE = new ArrayList<>();
        ArrayList<Motivo> motivosCAAT = new ArrayList<>();
        ArrayList<Motivo> motivosGC = new ArrayList<>();
        ArrayList<Motivo> motivosAux = new ArrayList<>();

        int distanciaMinimaTATA_BRE = 7; // Distancia minima desde caja TATA a motivo BRE aguas arriba.
        int distanciaMaximaTATA_BRE = 50; // Distancia minima desde caja TATA a motivo BRE aguas arriba.
        int distanciaMinimaTATA_CAAT = 150;//30; // Distancia minima desde caja TATA a motivo BRE aguas arriba.
        int distanciaMaximaTATA_CAAT = 800;//200; // Distancia minima desde caja TATA a motivo BRE aguas arriba.
        int distanciaMinimaCAAT_BRE = 150;//93; // Distancia minima desde caja TATA a motivo BRE aguas arriba.
        int distanciaMaximaCAAT_BRE = 800;//200; // Distancia minima desde caja TATA a motivo BRE aguas arriba.
        int distanciaMinimaCAAT_GC = 10;//93; // Distancia minima desde caja TATA a motivo BRE aguas arriba.
        int distanciaMaximaCAAT_GC = 50;//200; // Distancia minima desde caja TATA a motivo BRE aguas arriba.

        int menorCoordTATA, menorCoordBRE, menorBRE = -1;
        int distanciaGC_TATA;
        int distanciaCAAT_BRE, distanciaCAAT_TATA, distanciaBRE_TATA;
        int menorCoordCAAT, menorCAAT = -1;
        int menorCoordGC, menorGC = -1;

        String ftcomp;

        boolean hasConsense;
        boolean hasConsenseGCr;

        for (Motivo m : regionPromotora.getPromotor()) {

            for (factorTranscripcion ft : m.getFactores()) {

                ftcomp = ft.getID();

                for (String fTATA : factoresTATACorePromoter) {
                    String mS = m.getMotivo();

                    //hasConsense = Pattern.matches("[GC][TC][AT]T[AT][AT]A[AT][GA][GC][CG][GC][GC][GC][GC]", mS);
                    hasConsense = Pattern.matches("TATA[AT]A[AGT][AG]", mS);
                    // "TATA[AT]A[AGT][AG]",

                    if (hasConsense) {
                        m.setCore("TATA");
                    } else {                        // "AATGGGGGGGAA"
                        hasConsense = Pattern.matches("AATGGGGGGGAA", mS);
                        if (hasConsense) {
                            m.setCore("EIF4E-tata-like");
                        }/* else {
                         hasConsense = Pattern.matches("[TG]C[AT][GTC][TCA][CT][TCG][TC]", mS);
                         if (hasConsense) {
                         m.setCore("Inr-tata-like");
                         } else {
                         hasConsense = Pattern.matches("[AG]G[AT][CT][GAC]", mS);
                         if (hasConsense) {
                         m.setCore("DPE-tata-like");
                         }
                         }

                         }*/

                    }
                    if (consenso && !hasConsense) {
                        break;
                    }

                    if (ftcomp.indexOf(fTATA) != -1) {

                        if (!motivosTATA.contains(m)) {
                            if (!consenso) {
                                m.setCore("TATA");
                            }
                            motivosTATA.add(m);
                            break;
                        }
                    }
                }

                for (String fCAAT : factoresCAATCorePromoter) {

                    hasConsense = Pattern.matches("[TCA][CT][TC][AG][GA]CCA[AT][TA][CG][AG]", m.getMotivo());
                    //  "[TCA][CT][TC][AG][GA]CCA[AT][TA][CG][AG]"

                    if (consenso && !hasConsense) {
                        break;
                    }

                    if (ftcomp.indexOf(fCAAT) != -1) {

                        if (!motivosCAAT.contains(m)) {
                            m.setCore("CAAT");
                            motivosCAAT.add(m);
                            break;
                        }
                    }
                }

                for (String fBRE : factoresBRECorePromoter) {

                    hasConsense = Pattern.matches("[GC][GC][GA]CGCC", m.getMotivo());
                    //  "[GC][GC][GA]CGCC"

                    if (consenso && !hasConsense) {
                        break;
                    }

                    if (ftcomp.indexOf(fBRE) != -1) {
                        if (!motivosBRE.contains(m)) {
                            m.setCore("BRE");
                            motivosBRE.add(m);
                            break;
                        }
                    }
                }

                for (String fGC : factoresGCCorePromoter) {
                    // "[GT][GA]GGCG[GT][GA][GA][CT]"
                    hasConsense = Pattern.matches("[GT][GA]GGCG[GT][GA][GA][CT]", m.getMotivo());
                    hasConsenseGCr = Pattern.matches("[CT][GA][GA][GT]GCGG[GA][GT]", m.getMotivo());
                    // "[GT][GA]GGCG[GT][GA][GA][CT]"
                    //  "[CT][GA][GA][GT]GCGG[GA][GT]"

                    if (consenso && !(hasConsense || hasConsenseGCr)) {
                        break;
                    }

                    if (ftcomp.indexOf(fGC) != -1) {
                        if (!motivosGC.contains(m)) {
                            m.setCore("GC");
                            motivosGC.add(m);
                            break;
                        }
                    }
                }

            }

        }

        boolean corePromotersDefined = false;

        if (!motivosTATA.isEmpty() && !motivosBRE.isEmpty() && !motivosCAAT.isEmpty() && !motivosGC.isEmpty() && !corePromotersDefined) {
            //15

            for (Motivo cajaTATA : motivosTATA) {

                menorCoordTATA = cajaTATA.getCoordenadas()[0];
                Motivo breCajaTata = null;
                Motivo caatCajaBre = null;
                Motivo gcCajaTATA = null;
                for (Motivo bre : motivosBRE) {
                    menorCoordBRE = bre.getCoordenadas()[0];
                    if (menorCoordBRE < menorCoordTATA) {
                        distanciaBRE_TATA = menorCoordTATA - menorCoordBRE;
                        if ((distanciaMinimaTATA_BRE <= distanciaBRE_TATA) && (distanciaBRE_TATA <= distanciaMaximaTATA_BRE)) {
                            if (menorBRE == -1) {
                                menorBRE = menorCoordBRE;
                                breCajaTata = bre;
                            } else if (menorCoordBRE > menorBRE) {
                                menorBRE = menorCoordBRE;
                                breCajaTata = bre;
                            }
                        }
                    }
                }

                for (Motivo caat : motivosCAAT) {
                    menorCoordCAAT = caat.getCoordenadas()[0];
                    if (menorCoordCAAT < menorBRE) {
                        distanciaCAAT_BRE = menorBRE - menorCoordCAAT;
                        if ((distanciaMinimaCAAT_BRE <= distanciaCAAT_BRE) && (distanciaCAAT_BRE <= distanciaMaximaCAAT_BRE)) {
                            if (menorCAAT == -1) {
                                menorCAAT = menorCoordCAAT;
                                caatCajaBre = caat;
                            } else if (menorCoordCAAT > menorCAAT) {
                                menorCAAT = menorCoordCAAT;
                                caatCajaBre = caat;
                            }
                        }
                    }
                }

                for (Motivo gc : motivosGC) {
                    menorCoordGC = gc.getCoordenadas()[0];
                    distanciaGC_TATA = menorCoordTATA - menorCoordGC;
                    if ((distanciaMinimaTATA_CAAT <= distanciaGC_TATA) && (distanciaGC_TATA <= distanciaMaximaTATA_CAAT)) {
                        if (menorGC == -1) {
                            menorGC = menorCoordGC;
                            gcCajaTATA = gc;
                        } else if (menorCoordGC > menorGC) {
                            menorGC = menorCoordGC;
                            gcCajaTATA = gc;
                        }
                    }

                }

                ArrayList<Motivo> corePromoter = new ArrayList<>();

                if (caatCajaBre != null && breCajaTata != null && gcCajaTATA != null) {
                    corePromoter.add(gcCajaTATA);
                    corePromoter.add(caatCajaBre);
                    corePromoter.add(breCajaTata);
                    corePromoter.add(cajaTATA);
                    corePromoters.add(corePromoter);

                }
            }
            if (!corePromoters.isEmpty()) {
                corePromotersDefined = true;
                System.out.println("Esta region UTR5p contiene promotores tipo: 1 1 1 1" + " Gen:" + metaData.get_GenID());

            } else {

                motivosBRE.clear();
            }
        }

        if (!motivosTATA.isEmpty() && !motivosBRE.isEmpty() && !motivosCAAT.isEmpty() && motivosGC.isEmpty() && !corePromotersDefined) {
            //14

            for (Motivo cajaTATA : motivosTATA) {

                menorCoordTATA = cajaTATA.getCoordenadas()[0];
                Motivo breCajaTata = null;
                Motivo caatCajaBre = null;
                for (Motivo bre : motivosBRE) {
                    menorCoordBRE = bre.getCoordenadas()[0];
                    if (menorCoordBRE < menorCoordTATA) {
                        distanciaBRE_TATA = menorCoordTATA - menorCoordBRE;
                        if ((distanciaMinimaTATA_BRE <= distanciaBRE_TATA) && (distanciaBRE_TATA <= distanciaMaximaTATA_BRE)) {
                            if (menorBRE == -1) {
                                menorBRE = menorCoordBRE;
                                breCajaTata = bre;
                            } else if (menorCoordBRE > menorBRE) {
                                menorBRE = menorCoordBRE;
                                breCajaTata = bre;
                            }
                        }
                    }
                }

                for (Motivo caat : motivosCAAT) {
                    menorCoordCAAT = caat.getCoordenadas()[0];
                    if (menorCoordCAAT < menorBRE) {
                        distanciaCAAT_BRE = menorBRE - menorCoordCAAT;
                        if (distanciaCAAT_BRE <= distanciaMaximaCAAT_BRE) {
                            if (menorCAAT == -1) {
                                menorCAAT = menorCoordCAAT;
                                caatCajaBre = caat;
                            } else if (menorCoordCAAT > menorCAAT) {
                                menorCAAT = menorCoordCAAT;
                                caatCajaBre = caat;
                            }
                        }
                    }
                }

                ArrayList<Motivo> corePromoter = new ArrayList<>();

                if (caatCajaBre != null && breCajaTata != null) {
                    corePromoter.add(caatCajaBre);
                    corePromoter.add(breCajaTata);
                    corePromoter.add(cajaTATA);
                    corePromoters.add(corePromoter);
                }
            }

            if (!corePromoters.isEmpty()) {
                corePromotersDefined = true;
                System.out.println("Esta region UTR5p contiene promotores tipo: 1 1 1 0" + " Gen:" + metaData.get_GenID());
            } else {
                motivosBRE.clear();
            }
        }

        if (!motivosTATA.isEmpty() && !motivosBRE.isEmpty() && motivosCAAT.isEmpty() && !motivosGC.isEmpty() && !corePromotersDefined) {
            //13
            for (Motivo cajaTATA : motivosTATA) {

                menorCoordTATA = cajaTATA.getCoordenadas()[0];
                Motivo breCajaTATA = null;
                Motivo gcCajaTATA = null;
                for (Motivo bre : motivosBRE) {
                    menorCoordBRE = bre.getCoordenadas()[0];
                    if (menorCoordBRE < menorCoordTATA) {
                        distanciaBRE_TATA = menorCoordTATA - menorCoordBRE;
                        if (distanciaBRE_TATA <= distanciaMaximaTATA_BRE) {
                            if (menorBRE == -1) {
                                menorBRE = menorCoordBRE;
                                breCajaTATA = bre;
                            } else if (menorCoordBRE > menorBRE) {
                                menorBRE = menorCoordBRE;
                                breCajaTATA = bre;
                            }
                        }
                    }
                }

                for (Motivo gc : motivosGC) {
                    menorCoordGC = gc.getCoordenadas()[0];
                    distanciaGC_TATA = Math.abs(menorCoordTATA - menorCoordGC);
                    if ((distanciaMinimaTATA_CAAT <= distanciaGC_TATA) && (distanciaGC_TATA <= distanciaMaximaTATA_CAAT)) {
                        if (menorGC == -1) {
                            menorGC = menorCoordGC;
                            gcCajaTATA = gc;
                        } else if (menorCoordGC > menorGC) {
                            menorGC = menorCoordGC;
                            gcCajaTATA = gc;
                        }
                    }

                }
                ArrayList<Motivo> corePromoter = new ArrayList<>();

                if (breCajaTATA != null && gcCajaTATA != null) {
                    corePromoter.add(gcCajaTATA);
                    corePromoter.add(breCajaTATA);
                    corePromoter.add(cajaTATA);
                    corePromoters.add(corePromoter);

                }
            }

            if (!corePromoters.isEmpty()) {
                corePromotersDefined = true;
                System.out.println("Esta region UTR5p contiene promotores tipo: 1 1 0 1" + " Gen:" + metaData.get_GenID());
            } else {
                motivosBRE.clear();
            }
        }

        if (!motivosTATA.isEmpty() && !motivosBRE.isEmpty() && motivosCAAT.isEmpty() && motivosGC.isEmpty() && !corePromotersDefined) {
            //12

            for (Motivo cajaTATA : motivosTATA) {

                menorCoordTATA = cajaTATA.getCoordenadas()[0];
                Motivo breCajaTATA = null;
                for (Motivo bre : motivosBRE) {
                    menorCoordBRE = bre.getCoordenadas()[0];
                    if (menorCoordBRE < menorCoordTATA) {
                        distanciaBRE_TATA = menorCoordTATA - menorCoordBRE;
                        if (distanciaBRE_TATA <= distanciaMaximaTATA_BRE) {
                            if (menorBRE == -1) {
                                menorBRE = menorCoordBRE;
                                breCajaTATA = bre;
                            } else if (menorCoordBRE > menorBRE) {
                                menorBRE = menorCoordBRE;
                                breCajaTATA = bre;
                            }
                        }
                    }
                }
                ArrayList<Motivo> corePromoter = new ArrayList<>();
                if (breCajaTATA != null) {

                    corePromoter.add(breCajaTATA);
                    corePromoter.add(cajaTATA);
                    corePromoters.add(corePromoter);
                    corePromotersDefined = true;

                }
            }

            if (!corePromoters.isEmpty()) {
                corePromotersDefined = true;
                System.out.println("Esta region UTR5p contiene promotores tipo: 1 1 0 0" + " Gen:" + metaData.get_GenID());
            } else {
                motivosBRE.clear();
            }
        }

        if (!motivosTATA.isEmpty() && motivosBRE.isEmpty() && !motivosCAAT.isEmpty() && !motivosGC.isEmpty() && !corePromotersDefined) {
            //11

            for (Motivo cajaTATA : motivosTATA) {

                menorCoordTATA = cajaTATA.getCoordenadas()[0];
                Motivo caatCajaTATA = null;
                Motivo gcCajaTATA = null;
                for (Motivo caat : motivosCAAT) {
                    menorCoordCAAT = caat.getCoordenadas()[0];
                    if (menorCoordCAAT < menorCoordTATA) {
                        distanciaCAAT_TATA = menorCoordTATA - menorCoordCAAT;
                        if ((distanciaMinimaTATA_CAAT <= distanciaCAAT_TATA) && (distanciaCAAT_TATA <= distanciaMaximaTATA_CAAT)) {
                            if (menorCAAT == -1) {
                                menorCAAT = menorCoordCAAT;
                                caatCajaTATA = caat;
                            } else if (menorCoordCAAT > menorCAAT) {
                                menorCAAT = menorCoordCAAT;
                                caatCajaTATA = caat;
                            }
                        }
                    }
                }

                for (Motivo gc : motivosGC) {
                    menorCoordGC = gc.getCoordenadas()[0];
                    distanciaGC_TATA = Math.abs(menorCoordTATA - menorCoordGC);
                    if ((distanciaMinimaTATA_CAAT <= distanciaGC_TATA) && (distanciaGC_TATA <= distanciaMaximaTATA_CAAT)) {
                        if (menorGC == -1) {
                            menorGC = menorCoordGC;
                            gcCajaTATA = gc;
                        } else if (menorCoordGC > menorGC) {
                            menorGC = menorCoordGC;
                            gcCajaTATA = gc;
                        }
                    }

                }

                ArrayList<Motivo> corePromoter = new ArrayList<>();

                if (caatCajaTATA != null && gcCajaTATA != null) {
                    corePromoter.add(gcCajaTATA);
                    corePromoter.add(caatCajaTATA);
                    corePromoter.add(cajaTATA);
                    corePromoters.add(corePromoter);
                }

            }

            if (!corePromoters.isEmpty()) {
                corePromotersDefined = true;
                System.out.println("Esta region UTR5p contiene promotores tipo: 1 1 0 1" + " Gen:" + metaData.get_GenID());

            } else {
                motivosTATA.clear();
            }
        }

        if (!motivosTATA.isEmpty() && motivosBRE.isEmpty() && !motivosCAAT.isEmpty() && motivosGC.isEmpty() && !corePromotersDefined) {
            //10

            for (Motivo cajaTATA : motivosTATA) {

                menorCoordTATA = cajaTATA.getCoordenadas()[0];
                Motivo caatCajaTATA = null;
                for (Motivo caat : motivosCAAT) {
                    menorCoordCAAT = caat.getCoordenadas()[0];
                    if (menorCoordCAAT < menorCoordTATA) {
                        distanciaCAAT_TATA = menorCoordTATA - menorCoordCAAT;
                        if ((distanciaMinimaTATA_CAAT <= distanciaCAAT_TATA) && (distanciaCAAT_TATA <= distanciaMaximaTATA_CAAT)) {
                            if (menorCAAT == -1) {
                                menorCAAT = menorCoordCAAT;
                                caatCajaTATA = caat;
                            } else if (menorCoordCAAT > menorCAAT) {
                                menorCAAT = menorCoordCAAT;
                                caatCajaTATA = caat;
                            }
                        }
                    }
                }
                ArrayList<Motivo> corePromoter = new ArrayList<>();

                if (caatCajaTATA != null) {
                    corePromoter.add(caatCajaTATA);
                    corePromoter.add(cajaTATA);
                    corePromoters.add(corePromoter);
                }

            }

            if (!corePromoters.isEmpty()) {
                corePromotersDefined = true;
                System.out.println("Esta region UTR5p contiene promotores tipo: 1 0 1 0" + " Gen:" + metaData.get_GenID());

            } else {
                motivosTATA.clear();
            }

        }

        if (!motivosTATA.isEmpty() && motivosBRE.isEmpty() && motivosCAAT.isEmpty() && !motivosGC.isEmpty() && !corePromotersDefined) {
            // 9
            for (Motivo cajaTATA : motivosTATA) {

                menorCoordTATA = cajaTATA.getCoordenadas()[0];

                Motivo gcCajaTATA = null;

                for (Motivo gc : motivosGC) {
                    menorCoordGC = gc.getCoordenadas()[0];
                    distanciaGC_TATA = Math.abs(menorCoordTATA - menorCoordGC);
                    if ((distanciaMinimaTATA_CAAT <= distanciaGC_TATA) && (distanciaGC_TATA <= distanciaMaximaTATA_CAAT)) {
                        if (menorGC == -1) {
                            menorGC = menorCoordGC;
                            gcCajaTATA = gc;
                        } else if (menorCoordGC > menorGC) {
                            menorGC = menorCoordGC;
                            gcCajaTATA = gc;
                        }
                    }

                }

                ArrayList<Motivo> corePromoter = new ArrayList<>();

                if (gcCajaTATA != null) {
                    corePromoter.add(gcCajaTATA);
                    corePromoter.add(cajaTATA);
                    corePromoters.add(corePromoter);
                }
            }
            if (!corePromoters.isEmpty()) {
                corePromotersDefined = true;
                System.out.println("Esta region UTR5p contiene promotores tipo: 1 0 0 1" + " Gen:" + metaData.get_GenID());

            } else {
                motivosTATA.clear();
            }
        }

        if (!motivosTATA.isEmpty() && motivosBRE.isEmpty() && motivosCAAT.isEmpty() && motivosGC.isEmpty() && !corePromotersDefined) {
            // 8
            for (Motivo cajaTATA : motivosTATA) {
                ArrayList<Motivo> corePromoter = new ArrayList<>();
                corePromoter.add(cajaTATA);
                corePromoters.add(corePromoter);
            }

            corePromotersDefined = true;
            System.out.println("Esta region UTR5p contiene promotores tipo: 1 0 0 0" + " Gen:" + metaData.get_GenID());

        }

        if (motivosTATA.isEmpty() && !motivosBRE.isEmpty() && !motivosCAAT.isEmpty() && !motivosGC.isEmpty() && !corePromotersDefined) {
            // 7

            for (Motivo cajaBRE : motivosBRE) {

                menorCoordBRE = cajaBRE.getCoordenadas()[0];
                Motivo caatCajaBRE = null;
                Motivo gcCajaBRE = null;

                for (Motivo caat : motivosCAAT) {
                    menorCoordCAAT = caat.getCoordenadas()[0];
                    if (menorCoordCAAT < menorCoordBRE) {
                        distanciaCAAT_BRE = menorCoordBRE - menorCoordCAAT;
                        if ((distanciaMinimaCAAT_BRE <= distanciaCAAT_BRE) && (distanciaCAAT_BRE <= distanciaMaximaCAAT_BRE)) {
                            if (menorCAAT == -1) {
                                menorCAAT = menorCoordCAAT;
                                caatCajaBRE = caat;
                            } else if (menorCoordCAAT > menorCAAT) {
                                menorCAAT = menorCoordBRE;
                                caatCajaBRE = caat;
                            }
                        }
                    }
                }

                for (Motivo gc : motivosGC) {
                    menorCoordGC = gc.getCoordenadas()[0];
                    if (menorCoordGC < menorCoordBRE) {
                        distanciaCAAT_BRE = menorCoordBRE - menorCoordGC;
                        if ((distanciaMinimaCAAT_BRE <= distanciaCAAT_BRE) && (distanciaCAAT_BRE <= distanciaMaximaCAAT_BRE)) {
                            if (menorGC == -1) {
                                menorGC = menorCoordGC;
                                gcCajaBRE = gc;
                            } else if (menorCoordGC > menorGC) {
                                menorGC = menorCoordBRE;
                                gcCajaBRE = gc;
                            }
                        }
                    }
                }

                ArrayList<Motivo> corePromoter = new ArrayList<>();

                if (gcCajaBRE != null && caatCajaBRE != null) {
                    corePromoter.add(gcCajaBRE);
                    corePromoter.add(caatCajaBRE);
                    corePromoter.add(cajaBRE);
                    corePromoters.add(corePromoter);
                }

            }

            if (!corePromoters.isEmpty()) {
                corePromotersDefined = true;
                System.out.println("Esta region UTR5p contiene promotores tipo: 0 1 1 1");
            } else {
                motivosBRE.clear();
            }

        }

        if (motivosTATA.isEmpty() && !motivosBRE.isEmpty() && !motivosCAAT.isEmpty() && motivosGC.isEmpty() && !corePromotersDefined) {
            // 6

            for (Motivo cajaBRE : motivosBRE) {

                menorCoordBRE = cajaBRE.getCoordenadas()[0];
                Motivo caatCajaBRE = null;

                for (Motivo caat : motivosCAAT) {
                    menorCoordCAAT = caat.getCoordenadas()[0];
                    if (menorCoordCAAT < menorCoordBRE) {
                        distanciaCAAT_BRE = menorCoordBRE - menorCoordCAAT;
                        if ((distanciaMinimaCAAT_BRE <= distanciaCAAT_BRE) && (distanciaCAAT_BRE <= distanciaMaximaCAAT_BRE)) {
                            if (menorCAAT == -1) {
                                menorCAAT = menorCoordCAAT;
                                caatCajaBRE = caat;
                            } else if (menorCoordCAAT > menorCAAT) {
                                menorCAAT = menorCoordBRE;
                                caatCajaBRE = caat;
                            }
                        }
                    }
                }

                ArrayList<Motivo> corePromoter = new ArrayList<>();

                if (caatCajaBRE != null) {
                    corePromoter.add(caatCajaBRE);
                    corePromoter.add(cajaBRE);
                    corePromoters.add(corePromoter);
                }

            }

            if (!corePromoters.isEmpty()) {
                corePromotersDefined = true;
                System.out.println("Esta region UTR5p contiene promotores tipo: 0 1 1 0");
            } else {
                motivosBRE.clear();
            }

        }

        if (motivosTATA.isEmpty() && !motivosBRE.isEmpty() && motivosCAAT.isEmpty() && !motivosGC.isEmpty() && !corePromotersDefined) {
            // 5

            for (Motivo cajaBRE : motivosBRE) {
                menorCoordBRE = cajaBRE.getCoordenadas()[0];
                Motivo gcCajaBRE = null;
                for (Motivo gc : motivosGC) {
                    menorCoordGC = gc.getCoordenadas()[0];
                    if (menorCoordGC < menorCoordBRE) {
                        distanciaCAAT_BRE = menorCoordBRE - menorCoordGC;
                        if ((distanciaMinimaCAAT_BRE <= distanciaCAAT_BRE) && (distanciaCAAT_BRE <= distanciaMaximaCAAT_BRE)) {
                            if (menorGC == -1) {
                                menorGC = menorCoordGC;
                                gcCajaBRE = gc;
                            } else if (menorCoordGC > menorGC) {
                                menorGC = menorCoordBRE;
                                gcCajaBRE = gc;
                            }
                        }
                    }
                }

                ArrayList<Motivo> corePromoter = new ArrayList<>();

                if (gcCajaBRE != null) {
                    corePromoter.add(gcCajaBRE);
                    corePromoter.add(cajaBRE);
                    corePromoters.add(corePromoter);
                }
            }
            if (!corePromoters.isEmpty()) {
                corePromotersDefined = true;
                System.out.println("Esta region UTR5p contiene promotores tipo: 0 1 0 1");
            } else {
                motivosBRE.clear();
            }
        }

        if (motivosTATA.isEmpty() && !motivosBRE.isEmpty() && motivosCAAT.isEmpty() && motivosGC.isEmpty() && !corePromotersDefined) {
            // 4
            for (Motivo cajaBRE : motivosBRE) {
                ArrayList<Motivo> corePromoter = new ArrayList<>();
                corePromoter.add(cajaBRE);
                corePromoters.add(corePromoter);
            }

            corePromotersDefined = true;
            System.out.println("Esta region UTR5p contiene promotores tipo: 0 1 0 0");

        }

        if (motivosTATA.isEmpty() && motivosBRE.isEmpty() && !motivosCAAT.isEmpty() && !motivosGC.isEmpty() && !corePromotersDefined) {
            // 3
            int coordMenorCAAT, coordMenorGC, dif;

            for (Motivo cajaCAAT : motivosCAAT) {
                coordMenorCAAT = cajaCAAT.getCoordenadas()[0];

                for (Motivo gc : motivosGC) {

                    coordMenorGC = gc.getCoordenadas()[0];

                    dif = Math.abs(coordMenorCAAT - coordMenorGC);

                    if ((dif >= distanciaMinimaCAAT_GC) && (dif <= distanciaMaximaCAAT_GC)) {
                        ArrayList<Motivo> corePromoter = new ArrayList<>();
                        corePromoter.add(gc);
                        corePromoter.add(cajaCAAT);
                        corePromoters.add(corePromoter);

                    }
                }
            }

            if (!corePromoters.isEmpty()) {
                corePromotersDefined = true;
                System.out.println("Esta region UTR5p contiene promotores tipo: 0 0 1 1");
            } else {
                motivosCAAT.clear();
            }
        }

        if (motivosTATA.isEmpty() && motivosBRE.isEmpty() && !motivosCAAT.isEmpty() && motivosGC.isEmpty() && !corePromotersDefined) {
            // 2

            for (Motivo cajaCAAT : motivosCAAT) {
                ArrayList<Motivo> corePromoter = new ArrayList<>();
                corePromoter.add(cajaCAAT);
                corePromoters.add(corePromoter);
            }

            corePromotersDefined = true;
            System.out.println("Esta region UTR5p contiene promotores tipo: 0 0 1 0");

        }

        if (motivosTATA.isEmpty() && motivosBRE.isEmpty() && motivosCAAT.isEmpty() && !motivosGC.isEmpty() && !corePromotersDefined) {
            // 1

            for (Motivo cajaGC : motivosGC) {
                ArrayList<Motivo> corePromoter = new ArrayList<>();
                corePromoter.add(cajaGC);
                corePromoters.add(corePromoter);
            }

            corePromotersDefined = true;
            System.out.println("Esta region UTR5p contiene promotores tipo: 0 0 0 1");

        }

        return corePromoters;

    }

    /**
     * Metodo usado para la construccion de Regiones 5' para transcrito en
     * proceso. Ya se tiene un ORF y se desea completar el transcrito
     * correspondiente.
     */
    public ArrayList<ArrayList<Motivo>> definirUTR3p_CorePromoters(Region regionUTR3p, boolean consenso, Utilities metaData) throws Exception {

        ArrayList<ArrayList<Motivo>> corePromoters = new ArrayList<>(); // Contedra' todos los core promoters propuestos.

        String[] factoresPolyA_Site = {"CFI", "I factor (complement)", " IF", "C3b-INA", "C3b-inactivator", "FI", "KAF", "Konglutinogen-activating factor"};

        String[] factoresCPE = {"CPEB1", "CPEB", "FLJ13203"};
        String[] factoresPolyA_Signal = {"CPSF", "KIAA1367", "CLEAVAGE AND POLYADENYLATION SPECIFICITY FACTOR"};

        ArrayList<Motivo> motivosPolyA_Site = new ArrayList<>();
        ArrayList<Motivo> motivosPolyA_Signal = new ArrayList<>();
        ArrayList<Motivo> motivosCPE = new ArrayList<>();
        ArrayList<Motivo> motivosAux = new ArrayList<>();
        /*
         ArrayList<Motivo> motivosPolyA_Site = new ArrayList<>();
         ArrayList<Motivo> motivosPolyA_Signal = new ArrayList<>();
         ArrayList<Motivo> motivosCPE = new ArrayList<>();*/

        int distanciaMinimaPolyA_Site_PolyA_Signal = 10; // Distancia minima desde caja TATA a motivo BRE aguas arriba.
        int distanciaMaximaPolyA_Site_PolyA_Signal = 30; // Distancia minima desde caja TATA a motivo BRE aguas arriba.
        int distanciaMinimaPolyA_Site_CPE = 110; // Distancia minima desde caja TATA a motivo BRE aguas arriba.
        int distanciaMaximaPolyA_Site_CPE = 130; // Distancia minima desde caja TATA a motivo BRE aguas arriba.
        int distanciaMinimaCPE_PolyA_Signal = 100; // Distancia minima desde caja TATA a motivo BRE aguas arriba.
        int distanciaMaximaCPE_PolyA_Signal = 100; // Distancia minima desde caja TATA a motivo BRE aguas arriba.

        int menorCoordPolyA_Site, menorCoordPolyA_Signal, menorPolyA_Signal = -1;
        int distanciaPolyASignal_PolyA_Site, distanciaCPE_PolyA_Signal, distanciaCPE_PolyA_Site, menorCoordCPE, menorCPE = -1;
        boolean hasConsense;

        for (Motivo m : regionUTR3p.getPromotor()) {

            for (factorTranscripcion ft : m.getFactores()) {

                for (String fPolyA_Site : factoresPolyA_Site) {

                    if (ft.getID().indexOf(fPolyA_Site) != -1) {
                        if (!motivosPolyA_Site.contains(m)) {
                            motivosPolyA_Site.add(m);
                            break;
                        }
                    }
                }

                for (String fCPE : factoresCPE) {

                    hasConsense = Pattern.matches("TTTTT[T]+", m.getMotivo());

                    if (consenso && !hasConsense) {
                        break;
                    }
                    if (ft.getID().indexOf(fCPE) != -1) {
                        if (!motivosCPE.contains(m)) {
                            motivosCPE.add(m);
                            break;
                        }
                    }
                }

                for (String fPolyA_Signal : factoresPolyA_Signal) {

                    hasConsense = Pattern.matches("A[TA]TAAA", m.getMotivo());

                    if (consenso && !hasConsense) {
                        break;
                    }

                    if (ft.getID().indexOf(fPolyA_Signal) != -1) {
                        if (!motivosPolyA_Signal.contains(m)) {
                            motivosPolyA_Signal.add(m);
                            break;
                        }
                    }
                }

            }

        }

        boolean corePromotersDefined = false;

        if (!motivosPolyA_Site.isEmpty() && !motivosPolyA_Signal.isEmpty() && !motivosCPE.isEmpty() && !corePromotersDefined) {

            System.out.println("Esta region UTR3p contiene consensos tipo: 1 1 1" + " Gen:" + metaData.get_GenID());

            for (Motivo motivoPolyA_Site : motivosPolyA_Site) {

                menorCoordPolyA_Site = motivoPolyA_Site.getCoordenadas()[0];
                Motivo polyA_SignalCajaPolyA_Site = null;
                Motivo CPECajaPolyA_Signal = null;
                for (Motivo motivoPolyA_Signal : motivosPolyA_Signal) {
                    menorCoordPolyA_Signal = motivoPolyA_Signal.getCoordenadas()[0];
                    if (menorCoordPolyA_Signal < menorCoordPolyA_Site) {
                        distanciaPolyASignal_PolyA_Site = menorCoordPolyA_Site - menorCoordPolyA_Signal;
                        if (distanciaPolyASignal_PolyA_Site <= distanciaMaximaPolyA_Site_PolyA_Signal) {
                            if (menorPolyA_Signal == -1) {
                                menorPolyA_Signal = menorCoordPolyA_Signal;
                                polyA_SignalCajaPolyA_Site = motivoPolyA_Signal;
                            } else if (menorCoordPolyA_Signal > menorPolyA_Signal) {
                                menorPolyA_Signal = menorCoordPolyA_Signal;
                                polyA_SignalCajaPolyA_Site = motivoPolyA_Signal;
                            }
                        }
                    }
                }

                for (Motivo motivoCPE : motivosCPE) {
                    menorCoordCPE = motivoCPE.getCoordenadas()[0];
                    if (menorCoordCPE < menorPolyA_Signal) {
                        distanciaCPE_PolyA_Signal = menorPolyA_Signal - menorCoordCPE;
                        if (distanciaCPE_PolyA_Signal <= distanciaMaximaCPE_PolyA_Signal) {
                            if (menorCPE == -1) {
                                menorCPE = menorCoordCPE;
                                CPECajaPolyA_Signal = motivoCPE;
                            } else if (menorCoordCPE > menorCPE) {
                                menorCPE = menorCoordCPE;
                                CPECajaPolyA_Signal = motivoCPE;
                            }
                        }
                    }
                }

                ArrayList<Motivo> corePromoter = new ArrayList<>();

                if ((polyA_SignalCajaPolyA_Site != null) && (CPECajaPolyA_Signal != null)) {
                    corePromoter.add(CPECajaPolyA_Signal);
                    corePromoter.add(polyA_SignalCajaPolyA_Site);
                    corePromoter.add(motivoPolyA_Site);
                    corePromoters.add(corePromoter);

                }
            }
            if (!corePromoters.isEmpty()) {
                corePromotersDefined = true;
            } else {
                motivosAux = motivosCPE;
                motivosCPE = new ArrayList<>();
                System.out.println("Core promoters inexistentes para: " + metaData.get_GenID());
            }
        }

        /*

         if (!motivosPolyA_Site.isEmpty() && motivosPolyA_Signal.isEmpty() && !motivosCPE.isEmpty() && !corePromotersDefined) {

         System.out.println("Esta region UTR3p contiene promotores tipoCoreProm: 1 0 1");

         for (Motivo motivoPolyA_Site : motivosPolyA_Site) {

         menorCoordPolyA_Site = motivoPolyA_Site.getCoordenadas()[0];
         Motivo CPECajaPolyA_Site = null;
         for (Motivo motivoCPE : motivosCPE) {
         menorCoordCPE = motivoCPE.getCoordenadas()[0];
         if (menorCoordCPE < menorCoordPolyA_Site) {
         distanciaCPE_PolyA_Site = menorCoordPolyA_Site - menorCoordCPE;
         if ((distanciaMinimaPolyA_Site_CPE <= distanciaCPE_PolyA_Site) && (distanciaCPE_PolyA_Site <= distanciaMaximaPolyA_Site_CPE)) {
         if (menorCPE == -1) {
         menorCPE = menorCoordCPE;
         CPECajaPolyA_Site = motivoCPE;
         } else {
         if (menorCoordCPE > menorCPE) {
         menorCPE = menorCoordCPE;
         CPECajaPolyA_Site = motivoCPE;
         }
         }
         }
         }
         }
         ArrayList<Motivo> corePromoter = new ArrayList<>();

         if (CPECajaPolyA_Site != null) {
         corePromoter.add(CPECajaPolyA_Site);
         corePromoter.add(motivoPolyA_Site);
         corePromoters.add(corePromoter);
         }

         }

         if (!corePromoters.isEmpty()) {
         corePromotersDefined = true;
         } else {
         System.out.println("Core promoters inexistentes para: " + metaData.get_GenID());
         }


         }
         */
        if (!motivosPolyA_Site.isEmpty() && !motivosPolyA_Signal.isEmpty() && motivosCPE.isEmpty() && !corePromotersDefined) {

            System.out.println("Esta region UTR3p contiene consensos tipo: 1 1 0" + " Gen:" + metaData.get_GenID());

            for (Motivo motivoPolyA_Site : motivosPolyA_Site) {

                menorCoordPolyA_Site = motivoPolyA_Site.getCoordenadas()[0];
                Motivo PolyA_SignalCajaPolyA_Site = null;
                for (Motivo motivoPolyA_Signal : motivosPolyA_Signal) {
                    menorCoordPolyA_Signal = motivoPolyA_Signal.getCoordenadas()[0];
                    if (menorCoordPolyA_Signal < menorCoordPolyA_Site) {
                        distanciaPolyASignal_PolyA_Site = menorCoordPolyA_Site - menorCoordPolyA_Signal;
                        if ((distanciaMinimaPolyA_Site_PolyA_Signal <= distanciaPolyASignal_PolyA_Site) && (distanciaPolyASignal_PolyA_Site <= distanciaMaximaPolyA_Site_PolyA_Signal)) {
                            if (menorPolyA_Signal == -1) {
                                menorPolyA_Signal = menorCoordPolyA_Signal;
                                PolyA_SignalCajaPolyA_Site = motivoPolyA_Signal;
                            } else if (menorCoordPolyA_Signal > menorPolyA_Signal) {
                                menorPolyA_Signal = menorCoordPolyA_Signal;
                                PolyA_SignalCajaPolyA_Site = motivoPolyA_Signal;
                            }
                        }
                    }
                }
                ArrayList<Motivo> corePromoter = new ArrayList<>();
                if (PolyA_SignalCajaPolyA_Site != null) {

                    corePromoter.add(PolyA_SignalCajaPolyA_Site);
                    corePromoter.add(motivoPolyA_Site);
                    corePromoters.add(corePromoter);
                }

            }

            if (!corePromoters.isEmpty()) {
                corePromotersDefined = true;
            } else {
                motivosPolyA_Site.clear();
                motivosCPE = motivosAux;
                System.out.println("Core promoters inexistentes para: " + metaData.get_GenID());
            }

        }

        if (motivosPolyA_Site.isEmpty() && !motivosPolyA_Signal.isEmpty() && !motivosCPE.isEmpty() && !corePromotersDefined) {

            System.out.println("Esta region UTR3p contiene consensos tipo: 0 1 1" + " Gen:" + metaData.get_GenID());

            for (Motivo motivoPolyA_Signal : motivosPolyA_Signal) {

                menorCoordPolyA_Signal = motivoPolyA_Signal.getCoordenadas()[0];
                Motivo CPECajaPolyA_Signal = null;
                for (Motivo motivoCPE : motivosCPE) {
                    menorCoordCPE = motivoCPE.getCoordenadas()[0];
                    if (menorCoordCPE < menorCoordPolyA_Signal) {
                        distanciaCPE_PolyA_Signal = menorCoordPolyA_Signal - menorCoordCPE;
                        if (distanciaCPE_PolyA_Signal <= distanciaMaximaCPE_PolyA_Signal) {
                            if (menorCPE == -1) {
                                menorCPE = menorCoordCPE;
                                CPECajaPolyA_Signal = motivoCPE;
                            } else if (menorCoordCPE > menorCPE) {
                                menorCPE = menorCoordPolyA_Signal;
                                CPECajaPolyA_Signal = motivoCPE;
                            }
                        }
                    }
                }
                ArrayList<Motivo> corePromoter = new ArrayList<>();

                if (CPECajaPolyA_Signal != null) {
                    corePromoter.add(CPECajaPolyA_Signal);
                    corePromoter.add(motivoPolyA_Signal);
                    corePromoters.add(corePromoter);
                }

            }

            if (!corePromoters.isEmpty()) {
                corePromotersDefined = true;
            } else {
                motivosCPE.clear();
                System.out.println("Core promoters inexistentes para: " + metaData.get_GenID());
            }

        }
        /*
         if (motivosPolyA_Site.isEmpty() && !motivosPolyA_Signal.isEmpty() && motivosCPE.isEmpty() && !corePromotersDefined) {

         System.out.println("Esta region UTR3p contiene promotores tipoCoreProm: 0 1 0");

         for (Motivo motivoPolyA_Signal : motivosPolyA_Signal) {
         ArrayList<Motivo> corePromoter = new ArrayList<>();
         corePromoter.add(motivoPolyA_Signal);
         corePromoters.add(corePromoter);
         }
         if (!corePromoters.isEmpty()) {
         corePromotersDefined = true;
         } else {
         System.out.println("Core promoters inexistentes para: " + metaData.get_GenID());
         }
         }
         */
 /*
         if (motivosPolyA_Site.isEmpty() && motivosPolyA_Signal.isEmpty() && !motivosCPE.isEmpty() && !corePromotersDefined) {

         System.out.println("Esta region UTR3p contiene promotores tipoCoreProm: 0 0 1");

         for (Motivo motivoCPE : motivosCPE) {
         ArrayList<Motivo> corePromoter = new ArrayList<>();
         corePromoter.add(motivoCPE);
         corePromoters.add(corePromoter);
         }
         if (!corePromoters.isEmpty()) {
         corePromotersDefined = true;
         } else {
         System.out.println("Core promoters inexistentes para: UTR5p" + metaData.get_GenID());
         }
         }
         */
        if (motivosPolyA_Site.isEmpty() && !motivosPolyA_Signal.isEmpty() && motivosCPE.isEmpty() && !corePromotersDefined) {

            System.out.println("Esta region UTR3p contiene promotores tipo: 1 0 0");

            for (Motivo motivoPolyA_Site : motivosPolyA_Site) {
                ArrayList<Motivo> corePromoter = new ArrayList<>();
                corePromoter.add(motivoPolyA_Site);
                corePromoters.add(corePromoter);
            }
            if (!corePromoters.isEmpty()) {
                corePromotersDefined = true;
            } else {
                System.out.println("Core promoters inexistentes para: " + metaData.get_GenID());
            }
        }

        return corePromoters;

    }

    /**
     * Metodo usado para la construccion de Regiones 5' para transcrito en
     * proceso. Ya se tiene un ORF y se desea completar el transcrito
     * correspondiente.
     */
    public List<Integer> definirCoordPoliA(List<Information> regionPoliA) throws Exception {

        ArrayList<String> rPoliA = new ArrayList<>();
        for (Information i : regionPoliA) {
            rPoliA.add(i.toString());
            if (i.position < regionPoliA.size()) {
                rPoliA.add(",");
            }
        }

        String regionPolA = rPoliA.toString();
        regionPolA = regionPolA.replaceAll(" ", "");
        regionPolA = "region_polA(" + regionPolA + ").";

        File fileRegionUTR3p = new File("regionPoliA.pl");
        fileRegionUTR3p.delete();

        try (BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(fileRegionUTR3p, true), "UTF8"))) {
            out.write(regionPolA);
            out.write("\n");
            out.close();
        }

        MiddleWare middle = new MiddleWare();
        //middle.init("predictorPoliA.pl");
        middle.init("p_genes.pl");

        List<Integer> coordsPoliA = middle.consultPoliAs();

        return coordsPoliA;

    }

    /**
     * Metodo usado para la construccion de Regiones 5' para transcrito en
     * proceso. Ya se tiene un ORF y se desea completar el transcrito
     * correspondiente.
     */
    public List<Integer> definirSitiosParadaTranscripcion(List<Information> region) throws Exception {

        ArrayList<String> rParadT = new ArrayList<>();
        for (Information i : region) {
            rParadT.add(i.toString());
            if (i.position < region.size()) {
                rParadT.add(",");
            }
        }

        String regionParadaT = "region_parada_T([" + rParadT.toString() + "]).";

        File fileRegionUTR5p = new File("regionParadaT.pl");
        try (BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(fileRegionUTR5p, true), "UTF8"))) {
            out.write(regionParadaT);
            out.write("\n");
            out.close();
        }

        MiddleWare middle = new MiddleWare();
        middle.init("predictorParadaT.pl");

        List<Integer> coordsParadaT = middle.consultParadasT();

        return coordsParadaT;

    }

    //---------------------------------------
    //  </editor-fold>
    //---------------------------Private Methods-------------------------------- 
    // <editor-fold defaultstate="collapsed" desc="Private Methods">
    /**
     * Metodo que genera todos los intrones posibles encontrados en las listas
     * del GeneConstructor y las agrega a la lista de intrones de un Gene que
     * "no sera valido" y solo sera usado para conocer TODOS los intrones
     * POSIBLES
     */
    private Gene getPosibilities() throws Exception {
        Information start = constructor.getData(constructor.getAtg().get(0));
        Information end = constructor.getData(constructor.getStops().get(constructor.getStops().size() - 1));

        Gene posibilities = new Gene(start, end);

        for (Integer iniIntron : constructor.getGt()) {
            for (Integer finIntron : constructor.getAg()) {
                //finIntron = finIntron + 1;

                if (iniIntron < finIntron) {
                    int d = finIntron + 1 - iniIntron;

                    if (d > Model.minIntron && d < Model.maxIntron) {
                        start = constructor.getData(iniIntron);
                        end = constructor.getData(finIntron);

                        posibilities.addIntron(new Intron(start, end));
                    }
                }
            }
        }

        return posibilities;
    }

    //---------------------------------------
    /**
     * Metodo ITERATIVO que mezcla los intrones posibles en una "cola" de "lista
     * de intrones" donde cada lista sera una combinacion POSIBLE para una
     * lectura, pues se valida que pueda existir un exon entre ellos y demas
     * propiedades
     */
    private ArrayDeque<ArrayDeque<Intron>> iterativeMix(Gene possibilities) {

        ArrayDeque<ArrayDeque<Intron>> mixedIntrons = new ArrayDeque<>();

        ArrayDeque<Intron> introns = new ArrayDeque<>(possibilities.getIntrons());

        while (!introns.isEmpty()) {
            ArrayDeque<Intron> possibleMix = new ArrayDeque<>();
            possibleMix.add(introns.pollFirst());

            ArrayDeque<Intron> iteratorQueue = introns.clone();

            while (!iteratorQueue.isEmpty()) {
                Intron pos = iteratorQueue.pollFirst();

                int posS, possS, posE, possE;

                posS = pos.getStart().position;
                possS = possibleMix.peekLast().getStart().position;
                posE = pos.getEnd().position;
                possE = possibleMix.peekLast().getEnd().position;

                if (pos.getStart().position > possibleMix.peekLast().getStart().position
                        && pos.getEnd().position > possibleMix.peekLast().getEnd().position) {

                    int d = pos.getStart().position - possibleMix.peekLast().getEnd().position;

                    if (d >= Model.minExon && d <= Model.maxExon) {
                        possibleMix.add(pos);
                    }
                }
            }

            if (possibleMix.size() > 0) {

                mixedIntrons.add(possibleMix.clone());
                possibleMix.pollLast();
                while (!possibleMix.isEmpty()) {

                    mixedIntrons.add(possibleMix.clone());
                    possibleMix.pollLast();
                }
            }
        }

        return mixedIntrons;
    }

    //---------------------------------------
    /**
     * Metodo ITERATIVO que hace uso del metodo RECURSIVO para generar las
     * mezclas posibles de intrones que vienen en los intrones posibles
     */
    private ArrayDeque<ArrayDeque<Intron>> recursivelyMix(Gene possibilities) {
        ArrayDeque<ArrayDeque<Intron>> mixedIntrons = new ArrayDeque<>();
        int size = possibilities.getIntrons().size();

        for (int i = 0; i < size; i++) {
            mixedIntrons.addAll(recursivelyMix(null, i, 0, possibilities.getIntrons(), size, null));
        }

        return mixedIntrons;
    }

    //---------------------------------------
    /**
     * Metodo RECURSIVO que mezcla los intrones posibles en una "cola" de "lista
     * de intrones" donde cada lista sera una combinacion POSIBLE para una
     * lectura, pues se valida que pueda existir un exon entre ellos y demas
     * propiedades
     * <br/><br/>
     * <h1>"AUN NO FUNCIONAL"</h1>
     * <br/><br/>
     * Nota: Este metodo se desarrollo pues, la mezcla iterativa tiene casos que
     * no puede abarcar precisamente por ser iterativa, por lo que este metodo
     * es mas eficaz y debe ser usado en lugar del iterativo
     */
    private ArrayDeque<ArrayDeque<Intron>> recursivelyMix(Intron last, int pos, int level, List<Intron> posibilities, int size, ArrayDeque<Intron> paths) {
        ArrayDeque<ArrayDeque<Intron>> mixed = new ArrayDeque<>();

        if (paths == null) {
            last = posibilities.get(pos);
            level = 0;
            paths = new ArrayDeque<>();

            mixed.addAll(recursivelyMix(last, pos, level, posibilities, size, paths));
        } else {
            paths.add(last);

            while (++pos < size) {
                Intron possible = posibilities.get(pos);
                int d = possible.getStart().position - last.getEnd().position;

                if (d >= Model.minExon && d <= Model.maxExon) {
                    mixed.addAll(recursivelyMix(possible, pos, level + 1, posibilities, size, paths.clone()));
                }
            }

        }
        return mixed;
    }

    //---------------------------------------
    /**
     * Metodo que mezcla todas las combinaciones con los inicios y paradas
     * validas para que sea un gen y los agrega a la lista de genes(lectures) el
     * parametro "mixedIntrons" deberia ser la salida de alguno de los metodos
     * de combinacion (recursivelyMix, iterativeMix) para mayor efectividad
     */
    private void generateLectures(ArrayDeque<ArrayDeque<Intron>> mixedIntrons) throws Exception {
        while (!mixedIntrons.isEmpty()) {
            ArrayDeque<Intron> introns = mixedIntrons.poll();
            boolean shouldCheck = !(constructor.isWithoutStarts() || constructor.isWithoutStops());
            int coorLastIntron;
            coorLastIntron = introns.peekLast().getEnd().position;
            int coorFirstIntron;
            coorFirstIntron = introns.peekFirst().getStart().position;
            //System.out.println("coorFirstIntron: " + coorFirstIntron + "  coorLastIntron: " + coorLastIntron);
            for (Integer start : constructor.getAtg()) {
                int d = coorFirstIntron - start;

                if (d >= Model.minExon && d <= Model.maxExon) {
                    for (Integer end : constructor.getStops()) {

                        //System.out.println("coorStop: " + end.intValue());
                        d = (end + 1) - coorLastIntron;

                        if (d >= Model.minExon && d <= Model.maxExon) {
                            //  List<Information> geneData = constructor.getGeneData();

                            Information starE = constructor.getData(start);
                            Information finE = constructor.getData(end);

                            Gene lecture = new Gene(starE, finE);

                            boolean pass = true;
                            ArrayDeque<Intron> aux = introns.clone();
                            ArrayDeque<Intron> aux2 = introns.clone();
                            for (int i = 0; i < introns.size(); i++) {
                                if (!Restricciones.checkSizeIntron(aux.poll().getStart().position, aux2.poll().getEnd().position)) {
                                    pass = false;
                                }
                            }
                            if (pass) {
                                lecture.setIntrons(new ArrayList<>(introns));
                                lecture.inferExons(constructor);

                                int exonsLength = 0, inicio, fin;

                                for (int i = 0; i < lecture.getExons().size(); i++) {

                                    inicio = lecture.getExon(i).getStart().position;
                                    fin = lecture.getExon(i).getEnd().position;
                                    exonsLength = exonsLength + (fin - inicio + 1);

                                    if (!Restricciones.checkSizeExon(inicio, fin)) {
                                        pass = false;
                                        break;
                                    } else if (i == (lecture.getExons().size() - 1)) {
                                        if (!Restricciones.checkLength(exonsLength)) {
                                            pass = false;
                                        }
                                    }

                                }
                                if (pass) {

                                    Restricciones.addInnerInfo(lecture.getExons(), lecture.getIntrons(), constructor);
                                    if (!Restricciones.checkStops(lecture.getExons(), end)) { // no paradas intermedias?
                                        pass = false;
                                    }
                                }
                                if (pass) {
                                    String salida = "";
                                    this.lectures.add(lecture);
                                    System.out.println(salida + "\n");
                                } else {
                                    lecture = null;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    //*
    //---------------------------------------
    /**
     * Metodo que mezcla todas las combinaciones de intrones con los inicios y
     * paradas validas para que sea un gen y los agrega a la lista de
     * genes(lectures). El parametro "mixedIntrons" es la salida de alguno de
     * los metodos de combinacion (recursivelyMix, iterativeMix) para mayor
     * efectividad.
     *
     *
     */
    /*
     private void generateLectures(ArrayDeque<ArrayDeque<Intron>> mixedIntrons) throws Exception {
     while (!mixedIntrons.isEmpty()) {
     ArrayDeque<Intron> introns = mixedIntrons.poll();
     boolean shouldCheck = !(constructor.isWithoutStarts() || constructor.isWithoutStops());

     for (Integer start : constructor.getAtg()) {
     int d = introns.peekFirst().getStart().position - start;

     if (d >= Model.minExon && d <= Model.maxExon) {
     for (Integer end : constructor.getStops()) {
     d = end - introns.peekLast().getEnd().position;

     if (d >= Model.minExon && d <= Model.maxExon) {
     List<Information> geneData = constructor.getGeneData();

     Gene lecture = new Gene(constructor.getData(start), constructor.getData(end), shouldCheck, geneData);

     lecture.setIntrons(new ArrayList<>(introns));
     if (lecture.inferExons(geneData, !constructor.isWithoutStops(), shouldCheck)) {
     this.lectures.add(lecture);
     }

     //this.lectures.add(lecture);
     }
     }
     }
     }
     }

     }
    
     /*/
    public void lectsToString() {
        String out; //constructor.toString();

        int i = 0;
        for (Gene gene : lectures) {
            out = "";
            out += "Lectura #" + (++i) + "\n";
            //out += "GEN = " + gene.toString() + "\n";
            out += "--------\n";
            //out += "Intrones coordenadas" + "\n" + gene.getPositionsInfo(true) + "\n";
            //out += "Intrones DATA = " + gene.getStringInfo(true) + "\n";
            out += "Exones coordenadas" + "\n" + gene.getPositionsInfo(false).replace("(", "[").replace(")", "]") + "\n";

            out += "Exones DATA = " + gene.getStringInfo(false) + "\n";

            System.out.println(out);
        }

        //return out;
    }

    public int[] lectsToGTF(Utilities metaData, File salidaGTF, int[] globIDs, Region region) throws IOException {

        // Se actualizan los contadores para el seguimiento de IDs.
        int[] localIDs = new int[2];
        localIDs[0] = globIDs[0];
        localIDs[1] = globIDs[1];

        // Se imprime la cabecera del gen en proceso
        String inicioGlobalGen = metaData.get_InicioAbsoluto().toString();
        int coordenadaGlobGen = Integer.parseInt(inicioGlobalGen.replace("[", "").replace("]", ""));
        int referenciaGlobalCoords = coordenadaGlobGen - 600;
        //int referenciaGlobalCoords = 0;
        int coordenadaInicialGenGTF = referenciaGlobalCoords + constructor.getData(0).position + 1;
        int coordenadaFinalGenGTF = constructor.lastData() + referenciaGlobalCoords + 1;
        int coordIniTrans = 0, coordFinTrans = 0, coordIniExon = 0, coordFinExon = 0;

        metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tgene\t" + coordenadaInicialGenGTF + "\t" + coordenadaFinalGenGTF + "\t.\t" + metaData.get_hebra() + "\t.\tID=" + metaData.get_GenID().get(0) + ";Name=" + metaData.get_GenID().get(0), salidaGTF);

        // Se imprimen los transcritos como CDSs. Cada lectura es un CDS propuesto para el gen.
        int longExon, posicionIniExon, posicionFinExon, cuadratura = 0;
        String transcripID = "CDSILP.00" + localIDs[0]++;
        for (Gene gene : lectures) {
            //out=""; 

            // Se imprime cabecera de la lectura o CDS. Por ahora no se estan reportantando exones.
            coordIniTrans = gene.getStart().position + referenciaGlobalCoords + 1;
            Information i = gene.getEnd();
            int p = gene.getEnd().position;
            coordFinTrans = gene.getEnd().position + referenciaGlobalCoords + 3;
            metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\ttranscript\t" + coordIniTrans + "\t" + coordFinTrans + "\t.\t" + metaData.get_hebra() + "\t.\tID=" + transcripID + ";Name=" + transcripID + ";Parent=" + metaData.get_GenID().get(0), salidaGTF);

            //int contadorExones = gene.getExons().size();
            for (int contadorExones = 0; contadorExones < gene.getExons().size(); ++contadorExones) {

                posicionIniExon = gene.getExon(contadorExones).getStart().position + referenciaGlobalCoords + 1;
                posicionFinExon = gene.getExon(contadorExones).getEnd().position + referenciaGlobalCoords + 1;

                //longExon = posicionFinExon - posicionIniExon + 1;
                longExon = gene.getExon(contadorExones).getInnerInfo().size() + 2;
                if (contadorExones == 0) {
                    cuadratura = 0;
                } else {
                    cuadratura = (longExon % 3);

                }

                metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tCDS\t" + posicionIniExon + "\t" + posicionFinExon + "\t.\t" + metaData.get_hebra() + "\t" + cuadratura + "\tID=CDSILP0000" + localIDs[1] + ";Name=CDSILP0000" + localIDs[1] + ";Parent=" + transcripID, salidaGTF);

            }

        }

        if (this.lectures != null) {

            // Se reportan los motivos de lectura en curso.
            ArrayList<Motivo> promotor = region.getPromotor();

            for (Motivo motif : promotor) {

                int coordsIniMotif[] = motif.getCoordenadas();
                int coordIniMotif = coordsIniMotif[0] + referenciaGlobalCoords + 1;
                int coordFinMotif = coordsIniMotif[1] + referenciaGlobalCoords + 1;

                metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tTF_binding_site\t" + coordIniMotif + "\t" + coordFinMotif + "\t.\t" + metaData.get_hebra() + "\t" + "." + "\tID=tfbs0000" + localIDs[1] + ";Name=tfbs0000" + localIDs[1] + ";Parent=" + transcripID, salidaGTF);
                localIDs[1]++;

            }

            // Se reporta region UTR5' para la lectura en curso.
            int coordsPromo[] = region.getCoordenadasPromotor();
            int coordIniUTR5 = coordsPromo[1] + referenciaGlobalCoords + 1;
            int coordFinUTR5 = coordIniTrans - 1;

            if ((coordFinUTR5 - coordIniUTR5) > 0) { //Si hay distanciaPolyASignal_PolyA_Site entre el promotor y el atg de la lectura entonces se reporta.
                metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tUTR5\t" + coordIniUTR5 + "\t" + coordFinUTR5 + "\t.\t" + metaData.get_hebra() + "\t" + "." + "\tID=UTR5p0000" + localIDs[1] + ";Name=UTR5p0000" + localIDs[1] + ";Parent=" + transcripID, salidaGTF);
            }

            if (localIDs[0] == globIDs[0]) {
                System.out.println("No se han hallado lecturas o CDSs para:  " + metaData.get_GenID());
            } else {
                System.out.println("Se han generado " + (localIDs[0] - globIDs[0]) + " lecturas o CDSs para:  " + metaData.get_GenID());
            }

        }

        return localIDs;
    }

    public void lectsToGFF3(Utilities metaData, File salidaGTF, boolean reporteAbs, GenInformation genInformation, int numObjs, int numIter, String red, boolean complementarios) throws IOException, Exception {

        // Se imprime la cabecera del gen en proceso
        String inicioGlobalGen = metaData.get_InicioAbsoluto().toString();
        String finGlobalGen = metaData.get_FinAbsoluto().toString();
        int inicioG = Integer.parseInt(inicioGlobalGen.replace("[", "").replace("]", ""));
        int finG = Integer.parseInt(finGlobalGen.replace("[", "").replace("]", ""));

        String hebra = metaData.get_hebra();
        int coordenadaGlobGen, referenciaGlobalCoords;

        if (hebra.equalsIgnoreCase("+")) {
            coordenadaGlobGen = inicioG;
        } else {
            coordenadaGlobGen = finG;
        }

        if (reporteAbs) {
            referenciaGlobalCoords = coordenadaGlobGen;
        } else if (hebra.equalsIgnoreCase("+")) {
            referenciaGlobalCoords = 0;
        } else {
            referenciaGlobalCoords = finG - inicioG + 1;
        }

        //String genID = metaData.get_GenID().get(0);
        int cont_lects = 0;
        boolean utr5pdefined, utr3pdefined;

        for (Gene gene : lectures) {
            //*

            if (!gene.getUtr5ps().isEmpty()) {
                utr5pdefined = true;
            } else {
                utr5pdefined = false;
            }
            if (!gene.getUtr3p().isEmpty()) {
                utr3pdefined = true;
            } else {
                utr3pdefined = false;
            }

            String gen_ID = "gen-" + cont_lects;
            //*
            if (utr5pdefined && utr3pdefined) {// Se reporta el caso de transcritos con UTRs5p y UTRs3p
                cont_lects = reportarUTR5pUTR3p(gene, hebra, cont_lects, salidaGTF, referenciaGlobalCoords, gen_ID, metaData, genInformation, numObjs, numIter, red, complementarios);

            }/*/
            if (utr5pdefined && utr3pdefined) {// Se reporta el caso de transcritos con UTRs5p y no UTRs3p
                cont_lects = reportarUTR5p(gene, hebra, cont_lects, salidaGTF, referenciaGlobalCoords, gen_ID, metaData, genInformation, numObjs, numIter, red, complementarios);
            }//*
            if (!utr5pdefined && utr3pdefined) {// Se reporta el caso de transcritos con UTRs5p y no UTRs3p
                cont_lects = reportarUTR3p(gene, hebra, cont_lects, salidaGTF, referenciaGlobalCoords, gen_ID, metaData);
            }*/
            //*
            if (!utr5pdefined && !utr3pdefined) {// Se reporta el caso de transcritos con UTRs5p y no UTRs3p
                cont_lects = reportarCDS(gene, hebra, cont_lects, salidaGTF, referenciaGlobalCoords, gen_ID, metaData, genInformation, numObjs, numIter, red, complementarios);
            }//*/

            //this.constructORFListAbts(metaData, genInformation, gene, gene.getStart().position, gen_ID);
        }

    }

    public int reportarUTR5pUTR3p(Gene gene, String hebra, int cont_lects, File salidaGTF, int referenciaGlobalCoords, String gen_ID, Utilities metaData, GenInformation genInformation, int numObjs, int numIter, String red, boolean complementarios) throws IOException, Exception {

        // Se imprimen los transcritos.
        int longExon, coordIniTrans, coordFinTrans, posicionIniExon, posicionFinExon, cuadratura;

        // Se reporta el caso de transcritos con UTRs5p y UTRs3p para el ORF en curso.
        List<Gene> utr5ps = gene.getUtr5ps();
        List<UTR3p> utr3ps = gene.getUtr3p();
        int contador_UTR5ps = -1;
        int contador_UTR3ps = -1;

        int ref_mRNAs = 0, mRNAs;

        String utr5pFileAbtsID;

        int[] coordenadasGlobales = new int[2];
        coordenadasGlobales[0] = definirCoordenadaGlobalUTR5p(utr5ps);
        coordenadasGlobales[1] = definirCoordenadaGlobalUTR3p(utr3ps);

        int coordenadaInicialGenGFF = coordenadasGlobales[0];
        int coordenadaFinalGenGFF = coordenadasGlobales[1];

        if (hebra.equals("+")) {
            metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tgene\t" + (coordenadaInicialGenGFF + referenciaGlobalCoords + 1) + "\t" + (coordenadaFinalGenGFF + referenciaGlobalCoords + 1) + "\t.\t" + metaData.get_hebra() + "\t.\tID=" + gen_ID + ";Name=" + gen_ID, salidaGTF);
        } else {
            metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tgene\t" + (referenciaGlobalCoords - coordenadaFinalGenGFF) + "\t" + (referenciaGlobalCoords - coordenadaInicialGenGFF) + "\t.\t" + metaData.get_hebra() + "\t.\tID=" + gen_ID + ";Name=" + gen_ID, salidaGTF);
        }

        for (Gene utr5p : utr5ps) {
            contador_UTR5ps++;
            utr5pFileAbtsID = gen_ID + "-mRNA-" + ref_mRNAs + contador_UTR5ps;
            for (UTR3p utr3p : utr3ps) {
                contador_UTR3ps++;
                // Se imprime cabecera de la lectura o mRNA.
                coordIniTrans = utr5p.getStart().position;
                coordFinTrans = utr3p.getEnd().position;
                mRNAs = ref_mRNAs + contador_UTR5ps + contador_UTR3ps;
                String mRNA_ID = gen_ID + "-mRNA-" + mRNAs;

                if (hebra.equals("+")) {
                    metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tmRNA\t" + (coordIniTrans + referenciaGlobalCoords + 1) + "\t" + (coordFinTrans + referenciaGlobalCoords + 1) + "\t.\t" + metaData.get_hebra() + "\t.\tID=" + mRNA_ID + ";Name=" + mRNA_ID + ";Parent=" + gen_ID, salidaGTF);
                } else {
                    metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tmRNA\t" + (referenciaGlobalCoords - coordFinTrans) + "\t" + (referenciaGlobalCoords - coordIniTrans) + "\t.\t" + metaData.get_hebra() + "\t.\tID=" + mRNA_ID + ";Name=" + mRNA_ID + ";Parent=" + gen_ID, salidaGTF);
                }

                // Se reporta el UTR5p en curso.
                List<Motivo> corePromoter = utr5p.getPromoter();
                // Se reporta el UTR5p del transcrito en curso.
                int posIniUTR5p = coordIniTrans;
                int posFinUTR5p = utr5p.getEnd().position;
                String utr5pID = mRNA_ID + "-UTR5p-" + String.valueOf(contador_UTR5ps);

                if (hebra.equals("+")) {
                    metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tfive_prime_UTR\t" + (posIniUTR5p + referenciaGlobalCoords + 1) + "\t" + (posFinUTR5p + referenciaGlobalCoords + 1) + "\t.\t" + metaData.get_hebra() + "\t.\tID=" + utr5pID + ";Name=" + utr5pID + ";Parent=" + mRNA_ID, salidaGTF);
                } else {
                    metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tfive_prime_UTR\t" + (referenciaGlobalCoords - posFinUTR5p) + "\t" + (referenciaGlobalCoords - posIniUTR5p) + "\t.\t" + metaData.get_hebra() + "\t.\tID=" + utr5pID + ";Name=" + utr5pID + ";Parent=" + mRNA_ID, salidaGTF);
                }

                // Se reportan los motivos del core promoter contenidos en el UTR5p en curso.
                if (corePromoter != null) {
                    if (!corePromoter.isEmpty()) {
                        for (Motivo motivo : corePromoter) {

                            int posicionIniMotivo = motivo.getCoordenadas()[0];
                            int posicionFinMotivo = motivo.getCoordenadas()[1];
                            String motifID = gen_ID + motivo.getFactores().get(0).getID();

                            if (hebra.equals("+")) {
                                metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tTF_binding_site\t" + (posicionIniMotivo + referenciaGlobalCoords + 1) + "\t" + (posicionFinMotivo + referenciaGlobalCoords + 1) + "\t.\t" + metaData.get_hebra() + "\t" + "." + "\t" + "ID=" + motifID + ";Name=" + motifID + ";Parent=" + gen_ID, salidaGTF);
                            } else {
                                metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tTF_binding_site\t" + (referenciaGlobalCoords - posicionFinMotivo) + "\t" + (referenciaGlobalCoords - posicionIniMotivo) + "\t.\t" + metaData.get_hebra() + "\t" + "." + "\t" + "ID=" + motifID + ";Name=" + motifID + ";Parent=" + gen_ID, salidaGTF);
                            }
                        }
                    }
                }

                // Se reporta el UTR3p en curso.               
                int posIniUTR3p = utr3p.getStart().position;
                int posFinUTR3p = utr3p.getEnd().position;

                String utr3pID = mRNA_ID + "-UTR3p-" + String.valueOf(contador_UTR3ps);

                if (hebra.equals("+")) {
                    metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tthree_prime_UTR\t" + (posIniUTR3p + referenciaGlobalCoords + 1) + "\t" + (posFinUTR3p + referenciaGlobalCoords + 1) + "\t.\t" + metaData.get_hebra() + "\t.\tID=" + utr3pID + ";Name=" + utr3pID + ";Parent=" + mRNA_ID, salidaGTF);
                } else {
                    metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tthree_prime_UTR\t" + (referenciaGlobalCoords - posFinUTR3p) + "\t" + (referenciaGlobalCoords - posIniUTR3p) + "\t.\t" + metaData.get_hebra() + "\t.\tID=" + utr3pID + ";Name=" + utr3pID + ";Parent=" + mRNA_ID, salidaGTF);
                }

                // Se reportan los motivos de la region de poliadenilacion asociados al UTR3p en curso.
                if (utr3p.getPromoter() != null) {

                    List<Motivo> regionPolyA = utr3p.getPromoter();

                    if (!regionPolyA.isEmpty()) {
                        for (Motivo motivo : regionPolyA) {

                            int posicionIniMotivo = motivo.getCoordenadas()[0];
                            int posicionFinMotivo = motivo.getCoordenadas()[1];
                            String motifID = motivo.getFactores().get(0).getID();
                            String motifName = motivo.getFactores().get(0).getID();

                            if (hebra.equals("+")) {
                                metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tTF_binding_site\t" + (posicionIniMotivo + referenciaGlobalCoords + 1) + "\t" + (posicionFinMotivo + referenciaGlobalCoords + 1) + "\t.\t" + metaData.get_hebra() + "\t" + "." + "\t" + "ID=" + motifID + ";Name=" + motifName + ";Parent=" + gen_ID, salidaGTF);
                            } else {
                                metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tTF_binding_site\t" + (referenciaGlobalCoords - posicionFinMotivo) + "\t" + (referenciaGlobalCoords - posicionIniMotivo) + "\t.\t" + metaData.get_hebra() + "\t" + "." + "\t" + "ID=" + motifID + ";Name=" + motifName + ";Parent=" + gen_ID, salidaGTF);
                            }

                        }
                    }
                }

                // Se procede a reportar CDS.
                int numeCDSs = 0;
                int cantCDSs = gene.getExons().size();

                for (Exon cds : gene.getExons()) {

                    posicionIniExon = cds.getStart().position;
                    posicionFinExon = cds.getEnd().position;

                    longExon = cds.getInnerInfo().size();

                    if (cantCDSs == 1) {
                        cuadratura = 0;
                    } else {
                        cuadratura = (longExon % 3);

                    }
                    String cdsID = mRNA_ID + "-CDS-" + numeCDSs;
                    numeCDSs++;

                    if (hebra.equals("+")) {
                        metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tCDS\t" + (referenciaGlobalCoords + posicionIniExon + 1) + "\t" + (referenciaGlobalCoords + posicionFinExon + 1) + "\t.\t" + metaData.get_hebra() + "\t" + cuadratura + "\t" + "ID=" + cdsID + ";Name=" + cdsID + ";Parent=" + mRNA_ID, salidaGTF);
                    } else {
                        metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tCDS\t" + (referenciaGlobalCoords - posicionFinExon) + "\t" + (referenciaGlobalCoords - posicionIniExon) + "\t.\t" + metaData.get_hebra() + "\t" + cuadratura + "\t" + "ID=" + cdsID + ";Name=" + cdsID + ";Parent=" + mRNA_ID, salidaGTF);
                    }

                }

                // Se procede a reportar Exones.
                int numExons = 0;
                int totalExons = gene.getExons().size();

                for (Exon exon : gene.getExons()) {

                    if (numExons == 0 && totalExons > 1) {

                        posicionIniExon = coordIniTrans;
                        posicionFinExon = exon.getEnd().position;

                    } else if (numExons == 0 && totalExons == 1) {
                        posicionIniExon = coordIniTrans;
                        posicionFinExon = coordFinTrans;

                    } else if (numExons != 0 && numExons != totalExons - 1) {
                        posicionIniExon = exon.getStart().position;
                        posicionFinExon = exon.getEnd().position;
                    } else {
                        posicionIniExon = exon.getStart().position;
                        posicionFinExon = coordFinTrans;
                    }

                    String exonID = mRNA_ID + "-Exon-" + numExons;
                    numExons++;

                    if (hebra.equals("+")) {
                        metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\texon\t" + (referenciaGlobalCoords + posicionIniExon + 1) + "\t" + (referenciaGlobalCoords + posicionFinExon + 1) + "\t.\t" + metaData.get_hebra() + "\t" + "." + "\t" + "ID=" + exonID + ";Name=" + exonID + ";Parent=" + mRNA_ID, salidaGTF);
                    } else {
                        metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\texon\t" + (referenciaGlobalCoords - posicionFinExon) + "\t" + (referenciaGlobalCoords - posicionIniExon) + "\t.\t" + metaData.get_hebra() + "\t" + "." + "\t" + "ID=" + exonID + ";Name=" + exonID + ";Parent=" + mRNA_ID, salidaGTF);
                    }

                }

            }

            if (complementarios) {

                String genID = metaData.get_GenID().get(0);
                //String pathLocal = "salidas/estructuras" + "/" + red + "/" + genID;

                //String transFactFileID = gen_ID + ".tf";
                //File transFTfileID = new File(transFactFileID);
                String pathLocal = "salidas/estructuras";
                File path = new File(pathLocal);
                path.mkdir();

                String pathE = pathLocal + "/" + red;
                File pathEst = new File(pathE);
                pathEst.mkdir();

                String pathE1 = pathE + "/" + genID;
                File pathEst1 = new File(pathE1);
                pathEst1.mkdir();

                String pathEstruc = pathE1 + "/" + genID;
                File pathEstructura = new File(pathEstruc);
                pathEstructura.mkdir();

                String transFactFileID = pathEstructura + "/" + genID + ".tf";
                File transFTfileID = new File(transFactFileID);

                transFTfileID.delete();

                // Se mina el archivos de abstracts que describe los eventos de regulacion inherentes al UTR5p en proceso.
                this.listUTR5pHeader(metaData, gene, utr5p, gene.getStart().position, utr5pFileAbtsID, "5p3p", transFTfileID);
                this.constructORFListAbts(metaData, genInformation, utr5p, gene, utr5pFileAbtsID, utr5pFileAbtsID, numObjs, numIter);

            }

            ref_mRNAs = ref_mRNAs + contador_UTR3ps;
            contador_UTR3ps = -1;

        }

        // Se mina el archivos de abstracts que describe los eventos de regulacion inherentes al UTR5p en proceso.
        //this.constructORFListAbts(metaData, genInformation, gene, gene.getStart().position, gen_ID);
        return ++cont_lects;

    }

    private int[] definirCoordenadasGlobales(List<Gene> utr5ps, List<UTR3p> utr3ps) {

        int coordMenorMotivoUTRs5p = -1;
        int coordMayorMotivoUTRs3p = -1;
        Motivo menorMotivoUTR5p = null;
        Motivo mayorMotivoUTR3p = null;
        boolean utr5pTrue = false;
        boolean utr3pTrue = false;

        for (Gene utr5p : utr5ps) {
            if (!utr5p.getPromoter().isEmpty()) {

                menorMotivoUTR5p = utr5p.getPromoter().get(0);
                if (coordMenorMotivoUTRs5p == -1) {
                    coordMenorMotivoUTRs5p = menorMotivoUTR5p.getCoordenadas()[0];
                } else if (menorMotivoUTR5p.getCoordenadas()[0] < coordMenorMotivoUTRs5p) {
                    coordMenorMotivoUTRs5p = menorMotivoUTR5p.getCoordenadas()[0];
                }
            } else {
                System.out.println("Alerta: UTR5p sin region promotora");
            }
        }

        for (UTR3p utr3p : utr3ps) {

            if (!utr3p.getPromoter().isEmpty()) {
                int sizeUTR3p = utr3p.getPromoter().size();

                if (sizeUTR3p == 3) { // Motivo DSE
                    mayorMotivoUTR3p = utr3p.getPromoter().get(2);

                    if (coordMayorMotivoUTRs3p == -1) {
                        coordMayorMotivoUTRs3p = mayorMotivoUTR3p.getCoordenadas()[1];
                    } else if (mayorMotivoUTR3p.getCoordenadas()[1] > coordMayorMotivoUTRs3p) {
                        coordMayorMotivoUTRs3p = mayorMotivoUTR3p.getCoordenadas()[1];
                    }

                }

                if (sizeUTR3p == 2) {// Senal PolyA.
                    mayorMotivoUTR3p = utr3p.getPromoter().get(1);

                    if (coordMayorMotivoUTRs3p == -1) {
                        coordMayorMotivoUTRs3p = mayorMotivoUTR3p.getCoordenadas()[1];
                    } else if (mayorMotivoUTR3p.getCoordenadas()[1] > coordMayorMotivoUTRs3p) {
                        coordMayorMotivoUTRs3p = mayorMotivoUTR3p.getCoordenadas()[1];
                    }

                }

            } else {
                System.out.println("Alerta: UTR3p sin region promotora");
            }
        }


        /*
         if ((menorMotivoUTR5p != null) && (mayorMotivoUTR3p != null)) {
         coordenadaInicialGenGTF = coordMenorMotivoUTRs5p;
         coordenadaFinalGenGTF = coordMayorMotivoUTRs3p;
         }*/
        int[] coordendasGlobales = new int[2];
        coordendasGlobales[0] = coordMenorMotivoUTRs5p;
        coordendasGlobales[1] = coordMayorMotivoUTRs3p;
        return coordendasGlobales;

    }

    private int definirCoordenadaGlobalUTR3p(List<UTR3p> utr3ps) {

        int coordMayorMotivoUTRs3p = -1;

        Motivo mayorMotivoUTR3p = null;

        for (UTR3p utr3p : utr3ps) {

            if (utr3p.getPromoter() != null) {

                if (!utr3p.getPromoter().isEmpty()) {
                    int sizeUTR3p = utr3p.getPromoter().size();

                    if (sizeUTR3p == 3) { // Motivo DSE
                        mayorMotivoUTR3p = utr3p.getPromoter().get(2);

                        if (coordMayorMotivoUTRs3p == -1) {
                            coordMayorMotivoUTRs3p = mayorMotivoUTR3p.getCoordenadas()[1];
                        } else if (mayorMotivoUTR3p.getCoordenadas()[1] > coordMayorMotivoUTRs3p) {
                            coordMayorMotivoUTRs3p = mayorMotivoUTR3p.getCoordenadas()[1];
                        }

                    }

                    if (sizeUTR3p == 2) {// Senal PolyA.
                        mayorMotivoUTR3p = utr3p.getPromoter().get(1);

                        if (coordMayorMotivoUTRs3p == -1) {
                            coordMayorMotivoUTRs3p = mayorMotivoUTR3p.getCoordenadas()[1];
                        } else if (mayorMotivoUTR3p.getCoordenadas()[1] > coordMayorMotivoUTRs3p) {
                            coordMayorMotivoUTRs3p = mayorMotivoUTR3p.getCoordenadas()[1];
                        }

                    }

                } else {
                    coordMayorMotivoUTRs3p = utr3p.getEnd().position;
                }
            } else {
                coordMayorMotivoUTRs3p = utr3p.getEnd().position;
            }
        }

        return coordMayorMotivoUTRs3p;

    }

    private int definirCoordenadaGlobalUTR5p(List<Gene> utr5ps) {

        int coordMenorMotivoUTRs5p = -1;

        Motivo menorMotivoUTR5p = null;

        for (Gene utr5p : utr5ps) {
            if (utr5p.getPromoter() != null) {
                if (!utr5p.getPromoter().isEmpty()) {

                    menorMotivoUTR5p = utr5p.getPromoter().get(0);
                    if (coordMenorMotivoUTRs5p == -1) {
                        coordMenorMotivoUTRs5p = menorMotivoUTR5p.getCoordenadas()[0];
                    } else if (menorMotivoUTR5p.getCoordenadas()[0] < coordMenorMotivoUTRs5p) {
                        coordMenorMotivoUTRs5p = menorMotivoUTR5p.getCoordenadas()[0];
                    }
                } else {
                    System.out.println("Alerta: UTR5p sin region promotora");
                }
            }
            coordMenorMotivoUTRs5p = utr5p.getStart().position;

        }

        return coordMenorMotivoUTRs5p;

    }

    public int reportarUTR3p(Gene gene, String hebra, int cont_lects, File salidaGTF, int referenciaGlobalCoords, String gen_ID, Utilities metaData) throws IOException {

        // Se imprimen los transcritos.
        int longExon, coordIniTrans, coordFinTrans, posicionIniExon, posicionFinExon, cuadratura;

        List<UTR3p> utr3ps = gene.getUtr3p();

        int coordenadaInicialGenGTF = gene.getStart().position + referenciaGlobalCoords + 1;
        int coordenadaFinalGenGTF = definirCoordenadaGlobalUTR3p(utr3ps) + referenciaGlobalCoords + 1;

        metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tgene\t" + coordenadaInicialGenGTF + "\t" + coordenadaFinalGenGTF + "\t.\t" + metaData.get_hebra() + "\t.\tID=" + gen_ID + ";Name=" + gen_ID, salidaGTF);

        int mRNAs, ref_mRNAs = 0;
        int contador_UTR3ps = -1;

        // Se reporta el caso de transcritos con solo UTRs5p.
        for (UTR3p utr3p : utr3ps) {
            contador_UTR3ps++;
            // Se imprime cabecera de la lectura o mRNA.
            coordIniTrans = coordenadaInicialGenGTF;
            coordFinTrans = utr3p.getEnd().position + referenciaGlobalCoords + 1;
            mRNAs = contador_UTR3ps;
            String mRNA_ID = gen_ID + "-mRNA-" + mRNAs;
            metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tmRNA\t" + coordIniTrans + "\t" + coordFinTrans + "\t.\t" + metaData.get_hebra() + "\t.\tID=" + mRNA_ID + ";Name=" + mRNA_ID + ";Parent=" + gen_ID, salidaGTF);

            // Se reporta el UTR3p en curso.
            List<Motivo> regionPolyA = utr3p.getPromoter();
            // Se reporta el UTR3p del transcrito en curso.
            int posIniUTR3p = utr3p.getStart().position + referenciaGlobalCoords + 1;
            int posFinUTR3p = coordFinTrans;
            String utr3pID = mRNA_ID + "-UTR3p-" + String.valueOf(contador_UTR3ps);
            metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tthree_prime_UTR\t" + posIniUTR3p + "\t" + posFinUTR3p + "\t.\t" + metaData.get_hebra() + "\t.\tID=" + utr3pID + ";Name=" + utr3pID + ";Parent=" + mRNA_ID, salidaGTF);

            // Se reportan los motivos de la region de poliadenilacion asociados al UTR3p en curso.
            if (regionPolyA != null) {
                for (Motivo motivo : regionPolyA) {

                    int posicionIniMotivo = motivo.getCoordenadas()[0] + referenciaGlobalCoords + 1;
                    int posicionFinMotivo = motivo.getCoordenadas()[1] + referenciaGlobalCoords + 1;
                    String motifID = motivo.getFactores().get(0).getID();
                    String motifName = motivo.getFactores().get(0).getID();

                    metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tTF_binding_site\t" + posicionIniMotivo + "\t" + posicionFinMotivo + "\t.\t" + metaData.get_hebra() + "\t" + "." + "\t" + "ID=" + motifID + ";Name=" + motifName + ";Parent=" + gen_ID, salidaGTF);

                }
            }

            // Se procede a reportar CDS.
            int numeCDSs = 0;
            int cantCDSs = gene.getExons().size();

            for (Exon cds : gene.getExons()) {

                posicionIniExon = cds.getStart().position + referenciaGlobalCoords + 1;
                posicionFinExon = cds.getEnd().position + referenciaGlobalCoords + 1;

                //longExon = posicionFinExon - posicionIniExon + 1;
                longExon = cds.getInnerInfo().size() + 2;
                if (cantCDSs == 1) {
                    cuadratura = 0;
                } else {
                    cuadratura = (longExon % 3);

                }
                String cdsID = mRNA_ID + "-CDS-" + numeCDSs;
                numeCDSs++;
                metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tCDS\t" + posicionIniExon + "\t" + posicionFinExon + "\t.\t" + metaData.get_hebra() + "\t" + cuadratura + "\t" + "ID=" + cdsID + ";Name=" + cdsID + ";Parent=" + mRNA_ID, salidaGTF);

            }

            // Se procede a reportar Exones.
            int numExons = 0;
            int totalExons = gene.getExons().size();

            for (Exon exon : gene.getExons()) {

                if (numExons == 0 && totalExons > 1) {

                    posicionIniExon = coordIniTrans;
                    posicionFinExon = exon.getEnd().position + referenciaGlobalCoords + 1;

                } else if (numExons == 0 && totalExons == 1) {
                    posicionIniExon = coordIniTrans;
                    posicionFinExon = coordFinTrans;

                } else if (numExons != 0 && numExons != totalExons - 1) {
                    posicionIniExon = exon.getStart().position + referenciaGlobalCoords + 1;
                    posicionFinExon = exon.getEnd().position + referenciaGlobalCoords + 1;
                } else {
                    posicionIniExon = exon.getStart().position + referenciaGlobalCoords + 1;
                    posicionFinExon = coordFinTrans;
                }

                String exonID = mRNA_ID + "-Exon-" + numExons;
                numExons++;
                metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\texon\t" + posicionIniExon + "\t" + posicionFinExon + "\t.\t" + metaData.get_hebra() + "\t" + "." + "\t" + "ID=" + exonID + ";Name=" + exonID + ";Parent=" + mRNA_ID, salidaGTF);

            }

            //ref_mRNAs = ref_mRNAs + contador_UTR3ps;
        }

        return ++cont_lects;
    }

    public int reportarUTR5p(Gene gene, String hebra, int cont_lects, File salidaGTF, int referenciaGlobalCoords, String gen_ID, Utilities metaData, GenInformation genInformation, int numObjs, int numIter, String red, boolean complementarios) throws IOException, Exception {

        // Se imprimen los transcritos.
        int longExon, coordIniTrans, coordFinTrans, posicionIniExon, posicionFinExon, cuadratura = 0;

        List<Gene> utr5ps = gene.getUtr5ps();

        int coordenadaInicialGenGTF = definirCoordenadaGlobalUTR5p(utr5ps);
        int coordenadaFinalGenGTF = gene.getEnd().position;

        if (hebra.equals("+")) {
            metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tgene\t" + (coordenadaInicialGenGTF + referenciaGlobalCoords + 1) + "\t" + (coordenadaFinalGenGTF + referenciaGlobalCoords + 1) + "\t.\t" + metaData.get_hebra() + "\t.\tID=" + gen_ID + ";Name=" + gen_ID, salidaGTF);
        } else {
            metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tgene\t" + (referenciaGlobalCoords - coordenadaFinalGenGTF) + "\t" + (referenciaGlobalCoords - coordenadaInicialGenGTF) + "\t.\t" + metaData.get_hebra() + "\t.\tID=" + gen_ID + ";Name=" + gen_ID, salidaGTF);
        }

        int mRNAs, ref_mRNAs = 0;
        int contador_UTR5ps = -1;
        String utr5pFileAbtsID;
        // Se reporta el caso de transcritos con UTRs5p y no UTRs3p

        for (Gene utr5p : utr5ps) {
            contador_UTR5ps++;

            // Se imprime cabecera de la lectura o mRNA.
            coordIniTrans = utr5p.getStart().position;
            coordFinTrans = coordenadaFinalGenGTF;
            mRNAs = contador_UTR5ps;
            String mRNA_ID = gen_ID + "-mRNA-" + mRNAs;
            utr5pFileAbtsID = mRNA_ID + "-UTR5p-" + String.valueOf(contador_UTR5ps);
            //utr5pFileAbtsID = ORF_ID;

            if (hebra.equals("+")) {
                metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tmRNA\t" + (coordIniTrans + referenciaGlobalCoords + 1) + "\t" + (coordFinTrans + referenciaGlobalCoords + 1) + "\t.\t" + metaData.get_hebra() + "\t.\tID=" + mRNA_ID + ";Name=" + mRNA_ID + ";Parent=" + gen_ID, salidaGTF);
            } else {
                metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tmRNA\t" + (referenciaGlobalCoords - coordFinTrans) + "\t" + (referenciaGlobalCoords - coordIniTrans) + "\t.\t" + metaData.get_hebra() + "\t.\tID=" + mRNA_ID + ";Name=" + mRNA_ID + ";Parent=" + gen_ID, salidaGTF);
            }

            List<Motivo> corePromoter = utr5p.getPromoter();
            // Se reporta el UTR5p del transcrito en curso.
            int posIniUTR5p = utr5p.getStart().position;
            int posFinUTR5p = utr5p.getEnd().position + 1; // !!!!!! OJO que pasa que se requiere sumar 1????
            String utr5pID = mRNA_ID + "-UTR5p-" + String.valueOf(contador_UTR5ps);

            if (hebra.equals("+")) {
                metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tfive_prime_UTR\t" + (posIniUTR5p + referenciaGlobalCoords + 1) + "\t" + (posFinUTR5p + referenciaGlobalCoords + 1) + "\t.\t" + metaData.get_hebra() + "\t.\tID=" + utr5pID + ";Name=" + utr5pID + ";Parent=" + mRNA_ID, salidaGTF);
            } else {
                metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tfive_prime_UTR\t" + (referenciaGlobalCoords - posFinUTR5p) + "\t" + (referenciaGlobalCoords - posIniUTR5p) + "\t.\t" + metaData.get_hebra() + "\t.\tID=" + utr5pID + ";Name=" + utr5pID + ";Parent=" + mRNA_ID, salidaGTF);
            }

            // Se reportan los motivos del core promoter contenidos en el UTR5p en curso.
            if (corePromoter != null) {
                for (Motivo motivo : corePromoter) {

                    int posicionIniMotivo = motivo.getCoordenadas()[0];
                    int posicionFinMotivo = motivo.getCoordenadas()[1];
                    String motifID = gen_ID + "-" + motivo.getFactores().get(0).getID() + "-" + motivo.getCore();

                    if (hebra.equals("+")) {
                        metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tTF_binding_site\t" + (posicionIniMotivo + referenciaGlobalCoords + 1) + "\t" + (posicionFinMotivo + referenciaGlobalCoords + 1) + "\t.\t" + metaData.get_hebra() + "\t" + "." + "\t" + "ID=" + motifID + ";Name=" + motifID + ";Parent=" + gen_ID, salidaGTF);
                    } else {
                        metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tTF_binding_site\t" + (referenciaGlobalCoords - posicionFinMotivo) + "\t" + (referenciaGlobalCoords - posicionIniMotivo) + "\t.\t" + metaData.get_hebra() + "\t" + "." + "\t" + "ID=" + motifID + ";Name=" + motifID + ";Parent=" + gen_ID, salidaGTF);
                    }

                }
            }

            // Se procede a reportar CDS.
            int numeCDSs = 0;
            int cantCDSs = gene.getExons().size();

            for (Exon cds : gene.getExons()) {

                posicionIniExon = cds.getStart().position;
                posicionFinExon = cds.getEnd().position;

                //longExon = posicionFinExon - posicionIniExon + 1;
                longExon = cds.getInnerInfo().size();

                if (cantCDSs == 1) {
                    cuadratura = 0;
                } else {
                    cuadratura = (longExon % 3);

                }
                String cdsID = mRNA_ID + "-CDS-" + numeCDSs;
                numeCDSs++;

                if (hebra.equals("+")) {
                    metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tCDS\t" + (posicionIniExon + referenciaGlobalCoords + 1) + "\t" + (posicionFinExon + referenciaGlobalCoords + 1) + "\t.\t" + metaData.get_hebra() + "\t" + cuadratura + "\t" + "ID=" + cdsID + ";Name=" + cdsID + ";Parent=" + mRNA_ID, salidaGTF);
                } else {
                    metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tCDS\t" + (referenciaGlobalCoords - posicionFinExon) + "\t" + (referenciaGlobalCoords - posicionIniExon) + "\t.\t" + metaData.get_hebra() + "\t" + cuadratura + "\t" + "ID=" + cdsID + ";Name=" + cdsID + ";Parent=" + mRNA_ID, salidaGTF);
                }

            }

            // Se procede a reportar Exones.
            int numExons = 0;
            int totalExons = gene.getExons().size();

            for (Exon exon : gene.getExons()) {

                if (numExons == 0 && totalExons > 1) {

                    posicionIniExon = coordIniTrans;
                    posicionFinExon = exon.getEnd().position;

                } else if (numExons == 0 && totalExons == 1) {
                    posicionIniExon = coordIniTrans;
                    posicionFinExon = coordFinTrans;

                } else if (numExons != 0 && numExons != totalExons - 1) {
                    posicionIniExon = exon.getStart().position;
                    posicionFinExon = exon.getEnd().position;
                } else {
                    posicionIniExon = exon.getStart().position;
                    posicionFinExon = coordFinTrans;
                }

                String exonID = mRNA_ID + "-Exon-" + numExons;
                numExons++;

                if (hebra.equals("+")) {
                    metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\texon\t" + (posicionIniExon + referenciaGlobalCoords + 1) + "\t" + (posicionFinExon + referenciaGlobalCoords + 1) + "\t.\t" + metaData.get_hebra() + "\t" + "." + "\t" + "ID=" + exonID + ";Name=" + exonID + ";Parent=" + mRNA_ID, salidaGTF);
                } else {
                    metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\texon\t" + (referenciaGlobalCoords - posicionFinExon) + "\t" + (referenciaGlobalCoords - posicionIniExon) + "\t.\t" + metaData.get_hebra() + "\t" + "." + "\t" + "ID=" + exonID + ";Name=" + exonID + ";Parent=" + mRNA_ID, salidaGTF);
                }
            }
            /*
            if (complementarios) {
                String genID = metaData.get_GenID().get(0);
                //String pathLocal = "salidas/estructuras" + "/" + red + "/" + genID;
                String pathLocal = "salidas/estructuras";
                File path = new File(pathLocal);
                path.mkdir();

                String pathE = pathLocal + "/" + red;posFinUTR5p
                File pathEst = new File(pathE);
                pathEst.mkdir();

                String pathE1 = pathE + "/" + genID;
                File pathEst1 = new File(pathE1);
                pathEst1.mkdir();

                String pathEstruc = pathE1 + "/" + utr5pFileAbtsID;
                File pathEstructura = new File(pathEstruc);
                pathEstructura.mkdir();

                String transFactFileID = pathEstructura + "/" + utr5pFileAbtsID + ".tf";
                File transFTfileID = new File(transFactFileID);
                transFTfileID.delete();
                // Se mina el archivo de abstracts que describe los eventos de regulacion inherentes al UTR5p en proceso.

                this.listUTR5pHeader(metaData, gene, utr5p, gene.getStart().position, utr5pFileAbtsID, "5p", transFTfileID);
                this.constructORFListAbts(metaData, genInformation, utr5p, gene, utr5pFileAbtsID, pathEstruc, numObjs, numIter);

            }*/
            //ref_mRNAs = ref_mRNAs + ++contador_UTR5ps;
        }

        return ++cont_lects;

    }

    public int reportarCDS(Gene gene, String hebra, int cont_lects, File salidaGTF, int referenciaGlobalCoords, String gen_ID, Utilities metaData, GenInformation genInformation, int numObjs, int numIter, String red, boolean complementarios) throws IOException, Exception {

        // Se imprimen los transcritos compuestos solo de CDSs.
        int longExon, coordIniTrans, coordFinTrans, posicionIniExon, posicionFinExon, cuadratura;

        // Se reporta el caso de transcritos con UTRs5p y UTRs3p
        int contLecturas = cont_lects;

        // Se imprime cabecera de la lectura o mRNA.
        coordIniTrans = gene.getStart().position;
        coordFinTrans = gene.getEnd().position + 1;

        if (hebra.equals("+")) {
            metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tgene\t" + (coordIniTrans + referenciaGlobalCoords + 1) + "\t" + (coordFinTrans + referenciaGlobalCoords + 1) + "\t.\t" + metaData.get_hebra() + "\t.\tID=" + gen_ID + ";Name=" + gen_ID, salidaGTF);
        } else {
            metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tgene\t" + (referenciaGlobalCoords - coordFinTrans) + "\t" + (referenciaGlobalCoords - coordIniTrans) + "\t.\t" + metaData.get_hebra() + "\t.\tID=" + gen_ID + ";Name=" + gen_ID, salidaGTF);
        }

        String ORF_ID = gen_ID + "-ORF-" + "0";

        if (hebra.equals("+")) {
            metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tmRNA\t" + (coordIniTrans + referenciaGlobalCoords + 1) + "\t" + (coordFinTrans + referenciaGlobalCoords + 1) + "\t.\t" + metaData.get_hebra() + "\t.\tID=" + ORF_ID + ";Name=" + ORF_ID + ";Parent=" + gen_ID, salidaGTF);
        } else {
            metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tmRNA\t" + (referenciaGlobalCoords - coordFinTrans) + "\t" + (referenciaGlobalCoords - coordIniTrans) + "\t.\t" + metaData.get_hebra() + "\t.\tID=" + ORF_ID + ";Name=" + ORF_ID + ";Parent=" + gen_ID, salidaGTF);
        }

        // Se procede a reportar CDS.
        int numeCDSs = 0;

        for (Exon cds : gene.getExons()) {

            posicionIniExon = cds.getStart().position;
            posicionFinExon = cds.getEnd().position;

            //longExon = posicionFinExon - posicionIniExon + 1;
            longExon = cds.getInnerInfo().size();
            if (numeCDSs == 0) {
                cuadratura = 0;
            } else {
                cuadratura = (longExon % 3);

            }
            String cdsID = ORF_ID + "-CDS-" + numeCDSs;
            String exonID = ORF_ID + "-exon-" + numeCDSs;
            numeCDSs++;

            if (hebra.equals("+")) {
                metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tCDS\t" + (referenciaGlobalCoords + 1 + posicionIniExon) + "\t" + (posicionFinExon + referenciaGlobalCoords + 1) + "\t.\t" + metaData.get_hebra() + "\t" + cuadratura + "\t" + "ID=" + cdsID + ";Name=" + cdsID + ";Parent=" + ORF_ID, salidaGTF);
                metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\texon\t" + (referenciaGlobalCoords + 1 + posicionIniExon) + "\t" + (posicionFinExon + referenciaGlobalCoords + 1) + "\t.\t" + metaData.get_hebra() + "\t" + "." + "\t" + "ID=" + exonID + ";Name=" + exonID + ";Parent=" + ORF_ID, salidaGTF);
            } else {
                metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tCDS\t" + (referenciaGlobalCoords - posicionFinExon) + "\t" + (referenciaGlobalCoords - posicionIniExon) + "\t.\t" + metaData.get_hebra() + "\t" + cuadratura + "\t" + "ID=" + cdsID + ";Name=" + cdsID + ";Parent=" + ORF_ID, salidaGTF);
                metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\texon\t" + (referenciaGlobalCoords - posicionFinExon) + "\t" + (referenciaGlobalCoords - posicionIniExon) + "\t.\t" + metaData.get_hebra() + "\t" + "." + "\t" + "ID=" + exonID + ";Name=" + exonID + ";Parent=" + ORF_ID, salidaGTF);
            }

        }

        if (complementarios) {

            String genID = metaData.get_GenID().get(0);
            //String pathLocal = "salidas/estructuras" + "/" + red + "/" + genID;
            String pathLocal = "salidas/estructuras";
            File path = new File(pathLocal);
            path.mkdir();

            String pathE = pathLocal + "/" + red;
            File pathEst = new File(pathE);
            pathEst.mkdir();

            String pathE1 = pathE + "/" + genID;
            File pathEst1 = new File(pathE1);
            pathEst1.mkdir();

            String pathEstruc = pathE1 + "/" + ORF_ID;
            File pathEstructura = new File(pathEstruc);
            pathEstructura.mkdir();

            String transFactFileID = pathEstructura + "/" + ORF_ID + ".tf";
            File transFTfileID = new File(transFactFileID);
            transFTfileID.delete();

            // Se mina el archivos de abstracts que describe los eventos de regulacion inherentes al UTR5p en proceso.
            this.listUTR5pHeader(metaData, gene, gene, gene.getStart().position, ORF_ID, "CDS", transFTfileID);
            this.constructORFListAbts(metaData, genInformation, gene, gene, ORF_ID, pathEstruc, numObjs, numIter);

        }

        contLecturas++;
        return contLecturas;

    }

    public void lectsEnsemblEpd(Utilities metaData, File salidaEnsemblGenGTF, boolean reporteAbs, GenInformation genInformation) throws IOException {

        String ensemblIDEPD = metaData.get_GenID().get(0);
        String ensemblIDFile = ensemblIDEPD + ".gff3";

        File ensemblIDgff3 = new File(ensemblIDFile); // El reporte gff3 que viene de Ensembl para el gen con ID ensemblIDEPD.
        File ensemblIDsHUGOs = new File("ensemblIDs.txt"); // Los ensemblIDs y HUGO IDs en proceso.
        File promotoresUniEPD = new File("promotoresUniversales.txt"); // Los promotores reportados por EPD y sus TSS.
        File cajasPromotoresUniEPD = new File("PromotoresMotifsCoordinates.txt"); // Los promotores reportados por EPD y sus cajas.

        BufferedReader gff3EnProceso = new BufferedReader(new FileReader(ensemblIDgff3)); // Se lee el gff3 del gen ensemblIDEPD.
        BufferedReader ensemblIDsHUGO = new BufferedReader(new FileReader(ensemblIDsHUGOs)); // Ensembl y HUGO IDs.
        BufferedReader promotsUniEPD = new BufferedReader(new FileReader(promotoresUniEPD)); // Los promots desde EPD y sus TSS.
        BufferedReader cajasPromtsUniEPD = new BufferedReader(new FileReader(cajasPromotoresUniEPD)); // Los promots y sus cajas.

        // readers para cada archivo desde FindM.
        File fmTATA = new File("promotores_FM_TATA.fps"); //
        BufferedReader findM_TATA = new BufferedReader(new FileReader(fmTATA)); //
        File fmCAAT = new File("promotores_FM_CAAT.fps"); // 
        BufferedReader findM_CAAT = new BufferedReader(new FileReader(fmCAAT)); //
        File fmInr = new File("promotores_FM_Inr.fps"); // 
        BufferedReader findM_Inr = new BufferedReader(new FileReader(fmInr)); //
        File fmGC = new File("promotores_FM_GC.fps"); // 
        BufferedReader findM_GC = new BufferedReader(new FileReader(fmGC)); //

        // Obtenemos el promotor HUGO ID para el ensemblIDEPD en curso.
        boolean hugoIDdisponible = false;
        String hugoID = "";
        String ensemblID = "";
        int coordTssEnsembl = 0;

        while (ensemblIDsHUGO.ready()) { // Se obtiene el promoter HUGO ID para el Ensembl ID en curso.

            String ensIDH = ensemblIDsHUGO.readLine();
            String[] temp = ensIDH.split(" ");

            if (temp[0].equals(ensemblIDEPD)) {
                hugoID = temp[1];
                ensemblID = ensemblIDEPD;
                coordTssEnsembl = Integer.parseInt(temp[2]);
                hugoIDdisponible = true;
                break;
            }

        }

        // Se determina el EPD TSS para el Ensembl ID.
        String[] compsPromotsUniHUGO_ID = {};
        boolean cajasDisponibles = false;
        int coordTSS = 0; // Se define el TSS aqui porque el promotor en curso no necesariamente tiene caja Inr.
        String promtUniv;
        String promoID;

        if (hugoIDdisponible) {

            // Obtenemos la coordenada TSS desde el listado universal de promotores en EPD.
            while (promotsUniEPD.ready()) {

                promtUniv = promotsUniEPD.readLine();
                compsPromotsUniHUGO_ID = promtUniv.split(" ");
                promoID = compsPromotsUniHUGO_ID[3];

                if (promoID.equals(hugoID)) {
                    String temp1 = compsPromotsUniHUGO_ID[27].split(";")[0];
                    coordTSS = Integer.parseInt(temp1);
                    cajasDisponibles = true;
                    break;
                }

            }

        } else {
            System.out.println("Ese Ensembl ID no esta en la lista en proceso: " + ensemblIDEPD);
            System.exit(0);
        }

        // Se definen las coordenadas de las demas cajas respecto de la coordenada TSS.
        int coordRelTATA = 0;
        int coordRelCAAT = 0;
        int coordRelGC = 0;
        String[] compsCajasPromotsUniHUGO_ID = null;
        String lineCajasEPD;
        String hebra = metaData.get_hebra();
        String lineID;
        boolean cajasHugoID = false;

        if (cajasDisponibles) {

            while (cajasPromtsUniEPD.ready()) {

                lineCajasEPD = cajasPromtsUniEPD.readLine();
                compsCajasPromotsUniHUGO_ID = lineCajasEPD.split("\t");

                if (compsCajasPromotsUniHUGO_ID[0].equals(hugoID)) {
                    cajasHugoID = true;
                    break;
                }
            }

            if (cajasHugoID == false) {
                System.out.println("Inconsistencia: Para el Ensembl ID hay TSS pero no cajas: " + ensemblIDEPD);
            }

            // Determinamos las distancias relativas de cada caja para poteriormente reportar su ubicacion en el gff3 extendido.
            if (cajasHugoID) {
                if (compsCajasPromotsUniHUGO_ID[2].equals("1")) { // Hay caja Inr.

                    String idInr;
                    String lineInr;
                    String[] compsInr;
                    int coordInr;

                    while (findM_Inr.ready()) {

                        lineInr = findM_Inr.readLine();
                        compsInr = lineInr.split(" ");
                        lineID = compsInr[0];

                        if (lineID.equals("FP")) {
                            idInr = compsInr[3];
                            if (idInr.equals(hugoID)) {
                                String temp2 = compsInr[27].split(";")[0];
                                coordInr = Integer.parseInt(temp2);
                                if (!(coordInr == coordTSS)) {
                                    System.out.println("Ese Ensembl ID difiere en sus TSS EPD y FindM: " + ensemblIDEPD);
                                    System.out.println("Se reporta el TSS desde EPD");
                                }
                                break;
                            }
                        }
                    }
                }

                String idTATA;
                String lineTATA;
                String[] compsTATA;
                int coordTATA;

                if (compsCajasPromotsUniHUGO_ID[1].equals("1")) { // Hay caja TATA.

                    while (findM_TATA.ready()) {

                        lineTATA = findM_TATA.readLine();
                        compsTATA = lineTATA.split(" ");
                        lineID = compsTATA[0];

                        if (lineID.equals("FP")) {
                            idTATA = compsTATA[3];
                            if (idTATA.equals(hugoID)) { // Se determina distancia relativa caja TATA
                                String temp2 = compsTATA[27].split(";")[0];
                                coordTATA = Integer.parseInt(temp2);
                                if (hebra.equals("+")) {
                                    coordRelTATA = coordTSS - coordTATA;
                                } else {
                                    coordRelTATA = coordTATA - coordTSS;
                                }
                                break;
                            }

                        }
                    }
                }

                String idCAAT;
                String lineCAAT;
                String[] compsCAAT;
                int coordCAAT;

                if (compsCajasPromotsUniHUGO_ID[3].equals("1")) { // Hay caja CAAT.

                    while (findM_CAAT.ready()) {

                        lineCAAT = findM_CAAT.readLine();
                        compsCAAT = lineCAAT.split(" ");
                        lineID = compsCAAT[0];

                        if (lineID.equals("FP")) {
                            idCAAT = compsCAAT[3];
                            if (idCAAT.equals(hugoID)) { // Se determina distancia relativa caja CAAT
                                String temp2 = compsCAAT[27].split(";")[0];
                                coordCAAT = Integer.parseInt(temp2);
                                if (hebra.equals("+")) {
                                    coordRelCAAT = coordTSS - coordCAAT;
                                } else {
                                    coordRelCAAT = coordCAAT - coordTSS;
                                }
                                break;
                            }

                        }
                    }
                }

                String idGC;
                String lineGC;
                String[] compsGC;
                int coordGC;

                if (compsCajasPromotsUniHUGO_ID[4].equals("1")) { // Hay caja GC.

                    while (findM_GC.ready()) {

                        lineGC = findM_GC.readLine();
                        compsGC = lineGC.split(" ");
                        lineID = compsGC[0];

                        if (lineID.equals("FP")) {
                            idGC = compsGC[3];
                            if (idGC.equals(hugoID)) { // Se determina distancia relativa caja GC
                                String temp2 = compsGC[27].split(";")[0];
                                coordGC = Integer.parseInt(temp2);
                                if (hebra.equals("+")) {
                                    coordRelGC = coordTSS - coordGC;
                                } else {
                                    coordRelGC = coordGC - coordTSS;
                                }
                                break;
                            }

                        }
                    }
                }
            }

        } else {
            System.out.println("No hay cajas disponibles en EPD para promotor en el gen: " + ensemblIDEPD);
        }

        // Se reportan las cajas en el gff3 extendido.
        if (cajasHugoID) {
            reportarGFF3Ext(ensemblID, metaData, salidaEnsemblGenGTF, gff3EnProceso, compsCajasPromotsUniHUGO_ID, coordRelGC, coordRelCAAT, coordRelTATA, reporteAbs, coordTssEnsembl);
        } else {
            // Se lee el resto del reporte  gff3 del gen en curso y se agrega al gff3 extendido Ensembl EPD.
            String lineaGff3Curso;

            while (gff3EnProceso.ready()) {

                lineaGff3Curso = gff3EnProceso.readLine();
                metaData.guardar(lineaGff3Curso, salidaEnsemblGenGTF);

            }
        }

        findM_TATA.close();
        findM_CAAT.close();
        findM_Inr.close();
        findM_GC.close();

        gff3EnProceso.close();
        ensemblIDsHUGO.close();
        promotsUniEPD.close();
        cajasPromtsUniEPD.close();
    }

    public void reportarGFF3Ext(String ensemblID, Utilities metaData, File reporteEnsemblExt, BufferedReader gff3EnProceso, String[] compsCajasPromotsUniHUGO_ID, int coordRelGC, int coordRelCAAT, int coordRelTATA, boolean reporteAbs, int coordTssEnsembl) throws IOException {

        String inicioG, finG, coordenadaGlobGen, hebra = metaData.get_hebra();
        int coordTSSp1EPD = Integer.parseInt(metaData.get_InicioLocal().get(0)) - 1; // La coordenada TSS + 1 queda referida a index 0.
        int coordTSSp1Ensmbl = coordTssEnsembl - 1;
        int distRelEPDEnsembl;
        int referenciaGlobalCoords;

        inicioG = metaData.get_InicioAbsoluto().get(0).replaceAll("\\[", "").replaceAll("\\]", "");
        finG = metaData.get_FinAbsoluto().get(0).replaceAll("\\[", "").replaceAll("\\]", "");

        if (hebra.equalsIgnoreCase("+")) {
            coordenadaGlobGen = inicioG;
        } else {
            coordenadaGlobGen = finG;
        }

        if (reporteAbs) {
            referenciaGlobalCoords = Integer.parseInt(coordenadaGlobGen);
        } else if (hebra.equalsIgnoreCase("+")) {
            referenciaGlobalCoords = 0;
        } else {
            referenciaGlobalCoords = Integer.parseInt(finG) - Integer.parseInt(inicioG) + 1;
        }

        int coordGlobTATA;
        int coordGlobInr;
        int coordGlobCAAT;
        int coordGlobGC;
        String signo;

        if (hebra.equals("+")) {
            signo = "-";
            distRelEPDEnsembl = coordTSSp1EPD - coordTSSp1Ensmbl;
        } else {
            signo = "+";
            distRelEPDEnsembl = coordTSSp1Ensmbl - coordTSSp1EPD;
        }

        //Se reporta caja GC si la hay
        if (compsCajasPromotsUniHUGO_ID[4].equals("1")) {

            coordGlobGC = coordTSSp1EPD - 1 - coordRelGC;
            int coordIniGC = coordGlobGC - 6;
            int coordFinGC = coordGlobGC + 7;

            String motifID = ensemblID + "-GC-EPD-(" + distRelEPDEnsembl + ")(" + signo + (coordRelGC + 1 + 6) + ")";

            if (hebra.equals("+")) {
                metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tTF_binding_site\t" + (coordIniGC + referenciaGlobalCoords + 1) + "\t" + (coordFinGC + referenciaGlobalCoords + 1) + "\t.\t" + metaData.get_hebra() + "\t" + "." + "\t" + "ID=" + motifID + ";Name=" + motifID + ";Parent=" + ensemblID, reporteEnsemblExt);
            } else {
                metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tTF_binding_site\t" + (referenciaGlobalCoords - coordFinGC) + "\t" + (referenciaGlobalCoords - coordIniGC) + "\t.\t" + metaData.get_hebra() + "\t" + "." + "\t" + "ID=" + motifID + ";Name=" + motifID + ";Parent=" + ensemblID, reporteEnsemblExt);
            }
        }

        //Se reporta caja CAAT si la hay
        if (compsCajasPromotsUniHUGO_ID[3].equals("1")) {

            coordGlobCAAT = coordTSSp1EPD - 1 - coordRelCAAT;
            int coordIniCAAT = coordGlobCAAT - 7;
            int coordFinCAAT = coordGlobCAAT + 4;

            String motifID = ensemblID + "-CAAT-EPD-(" + distRelEPDEnsembl + ")(" + signo + (coordRelCAAT + 1 + 7) + ")";

            if (hebra.equals("+")) {
                metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tTF_binding_site\t" + (coordIniCAAT + referenciaGlobalCoords + 1) + "\t" + (coordFinCAAT + referenciaGlobalCoords + 1) + "\t.\t" + metaData.get_hebra() + "\t" + "." + "\t" + "ID=" + motifID + ";Name=" + motifID + ";Parent=" + ensemblID, reporteEnsemblExt);
            } else {
                metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tTF_binding_site\t" + (referenciaGlobalCoords - coordFinCAAT) + "\t" + (referenciaGlobalCoords - coordIniCAAT) + "\t.\t" + metaData.get_hebra() + "\t" + "." + "\t" + "ID=" + motifID + ";Name=" + motifID + ";Parent=" + ensemblID, reporteEnsemblExt);
            }
        }

        //Se reporta caja TATA si la hay
        if (compsCajasPromotsUniHUGO_ID[1].equals("1")) {

            coordGlobTATA = coordTSSp1EPD - 1 - coordRelTATA; // -1 porque la distancia TATA esta reportada respecto del TSS.
            int coordIniTATA = coordGlobTATA - 3;
            int coordFinTATA = coordGlobTATA + 11;
            //String motifID = ensemblID + "-TATA-EPD-"+(coordRelTATA+4);

            String motifID = ensemblID + "-TATA-EPD-(" + distRelEPDEnsembl + ")(" + signo + (coordRelTATA + 1 + 3) + ")";

            if (hebra.equals("+")) {
                metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tTF_binding_site\t" + (coordIniTATA + referenciaGlobalCoords + 1) + "\t" + (coordFinTATA + referenciaGlobalCoords + 1) + "\t.\t" + metaData.get_hebra() + "\t" + "." + "\t" + "ID=" + motifID + ";Name=" + motifID + ";Parent=" + ensemblID, reporteEnsemblExt);
            } else {
                metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tTF_binding_site\t" + (referenciaGlobalCoords - coordFinTATA) + "\t" + (referenciaGlobalCoords - coordIniTATA) + "\t.\t" + metaData.get_hebra() + "\t" + "." + "\t" + "ID=" + motifID + ";Name=" + motifID + ";Parent=" + ensemblID, reporteEnsemblExt);
            }
        }

        //Se reporta caja Inr si la hay
        if (compsCajasPromotsUniHUGO_ID[2].equals("1")) {

            coordGlobInr = coordTSSp1EPD - 1;
            int coordIniInr = coordTSSp1EPD - 3;
            int coordFinInr = coordTSSp1EPD + 5;

            String motifID = ensemblID + "-Inr-EPD-(" + distRelEPDEnsembl + ")(" + signo + (1 + 3) + ")";

            if (hebra.equals("+")) {
                metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tTF_binding_site\t" + (coordIniInr + referenciaGlobalCoords + 1) + "\t" + (coordFinInr + referenciaGlobalCoords + 1) + "\t.\t" + metaData.get_hebra() + "\t" + "." + "\t" + "ID=" + motifID + ";Name=" + motifID + ";Parent=" + ensemblID, reporteEnsemblExt);
            } else {
                metaData.guardar(metaData.get_Cromosoma().get(0) + "\tPredictorILP\tTF_binding_site\t" + (referenciaGlobalCoords - coordFinInr) + "\t" + (referenciaGlobalCoords - coordIniInr) + "\t.\t" + metaData.get_hebra() + "\t" + "." + "\t" + "ID=" + motifID + ";Name=" + motifID + ";Parent=" + ensemblID, reporteEnsemblExt);
            }
        }

        // Se lee el resto del reporte  gff3 del gen en curso y se agrega al gff3 extendido Ensembl EPD.
        String lineaGff3Curso;

        while (gff3EnProceso.ready()) {

            lineaGff3Curso = gff3EnProceso.readLine();
            metaData.guardar(lineaGff3Curso, reporteEnsemblExt);

        }

    }

//  </editor-fold>
//---------------------------Override Methods------------------------------- 
// <editor-fold defaultstate="collapsed" desc="Override Methods">
    @Override
    public String toString() {
        String out = constructor.toString();

        int i = 0;
        for (Gene gene : lectures) {
            // out += "\n----------------------------Lectura #" + (++i) + "----------------------------------\n";
            // out += "GEN = " + gene.toString() + "\n";
            // out += "----------------------------------------------------\n";
            out += "N I N  " + gene.getPositionsInfo(true) + "\n";
            // out += "Intrones DATA = " + gene.getStringInfo(true) + "\n";
            // out += "----------------------------------------------------\n";
            out += "N E N  " + gene.getPositionsInfo(false).replace("(", "[").replace(")", "]") + "\n";
            // out += "Exones DATA = " + gene.getStringInfo(false) + "\n";
        }

        return out;
    }
    //---------------------------------------
    //  </editor-fold>
}
