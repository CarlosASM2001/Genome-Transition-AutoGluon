/*
    Gene.java


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

package gene.feature;

import static gene.feature.Gene.endPattern;
import static gene.feature.Gene.startPattern;
import gene.information.GeneConstructor;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.regex.Pattern;
import pipeline.Motivo;

/**
 * Clase que extiende de GenePart y especifica un GEN como tal, posee una lista
 * de intrones y una de exones que seran todos los valores correspondientes a un
 * gen valido, por lo tanto un objeto de esta clase, deberia ser usado como una
 * LECTURA VALIDA
 */
public class Gene extends GenePart {
    //---------------------------Static Constants-------------------------------
    // <editor-fold desc="Static Constants">

    public static final Pattern startPattern = Pattern.compile(Model.ATG);
    public static final Pattern endPattern = Pattern.compile(Model.stops[Model.TAA] + "|" + Model.stops[Model.TAG] + "|" + Model.stops[Model.TGA]);
    //---------------------------------------
    //  </editor-fold>
    //---------------------------Private Attributes-----------------------------
    // <editor-fold desc="Private Attributes">
    private List<Motivo> promoter;
    private String core;

    public String getCore() {
        return core;
    }

    public void setCore(String core) {
        this.core = core;
    }

    public List<Motivo> getPromoter() {
        return promoter;
    }
    private List<Intron> introns;
    private List<Exon> exons;
    private List<Gene> utr5ps;
    private List<UTR3p> utr3p;

    public void setPromotor(List<Motivo> promotor) {
        this.promoter = promotor;
    }

    public void setStart(Information start) {
        this.start = start;
    }

    public void setEnd(Information end) {
        this.end = end;
    }

    public void setInnerInfo(List<Information> innerInfo) {
        this.innerInfo = innerInfo;
    }
    
    public static Pattern getStartPattern() {
        return startPattern;
    }

    public static Pattern getEndPattern() {
        return endPattern;
    }

    public boolean isCheckEdges() {
        return checkEdges;
    }

    public void setCheckEdges(boolean checkEdges) {
        this.checkEdges = checkEdges;
    }
    private boolean checkEdges;

    //---------------------------------------
    //  </editor-fold>
    //---------------------------Constructors-----------------------------------
    // <editor-fold defaultstate="collapsed" desc="Constructors">
    /**
     * Se requiere solo la informacion de donde comienza y donde termina el gen
     * es decir el "atg" y cualquier Parada (taa,tag,tga), sin comprobacion de
     * correspondencia valida
     */
    public Gene(Information start, Information end) {
        super(start, end);
        this.innerInfo = new ArrayList<>();
        this.checkEdges = false;
        this.utr3p = new ArrayList<>();
        this.utr5ps = new ArrayList<>();        
    }
    
    public Gene(Information start, Information end, List<Information> geneData) {
        super(start, end);
        this.innerInfo = geneData;
        this.checkEdges = false;
        this.utr3p = new ArrayList<>();
        this.utr5ps = new ArrayList<>();        
    }

    //---------------------------------------
    /**
     * Se requiere la informacion de donde comienza y donde termina el gen y
     * recibe un boolean para saber si comprobar o no los bordes con sus
     * correspondientes valores atg(Inicio) taa,tag,tga(Paradas)
     */
    public Gene(Information start, Information end, boolean checkEdges, List<Information> geneData) throws Exception {
        super(start, end);
        this.innerInfo = new ArrayList<>();
        this.checkEdges = checkEdges;
        this.utr3p = new ArrayList<>();
        this.utr5ps = new ArrayList<>();  
        

        if (this.checkEdges) {
            int is = this.start.position;
            String atg = this.start.toString() + geneData.get(++is).toString() + geneData.get(++is).toString();
            int ie = this.end.position;
            String stop = this.end.toString() + geneData.get(++ie).toString() + geneData.get(++ie).toString();

            if (!startPattern.matcher(atg).find() || !endPattern.matcher(stop).find()) {
                is = this.start.position;
                ie = this.end.position;

                throw new Exception("GEN INVALIDO, (" + is + ":" + ie + ")[" + atg + ":" + stop + "]"
                        + " NO CORRESPONDE a una estructura de gen VALIDA "
                        + "[" + startPattern.pattern() + ":" + endPattern.pattern() + "]");
            }

            this.end = geneData.get(ie);
        }
    }
    //  </editor-fold>
    //---------------------------Getters---------------------------------------- 
    // <editor-fold defaultstate="collapsed" desc="Getters">

    public List<Intron> getIntrons() {
        return introns;
    }

    //---------------------------------------
    public List<Exon> getExons() {
        return exons;
    }

    //---------------------------------------
    public Exon getExon(int i) {
        return exons.get(i);
    }

    //---------------------------------------
    public Intron getIntron(int i) {
        return introns.get(i);
    }

    //---------------------------------------
    public List<Gene> getUtr5ps() {
        return utr5ps;
    }
    //---------------------------------------

    public List<UTR3p> getUtr3p() {
        return utr3p;
    }

    //  </editor-fold>
    //---------------------------Setters---------------------------------------- 
    // <editor-fold defaultstate="collapsed" desc="Setters">
    public void setIntrons(List<Intron> introns) {
        this.introns = introns;
    }

    //---------------------------------------
    public void setExons(List<Exon> exons) {
        this.exons = exons;
    }

    public void setUtr5ps(List<Gene> utr5ps) {
        this.utr5ps = utr5ps;
    }

    public void setUtr3p(List<UTR3p> utr3p) {
        this.utr3p = utr3p;
    }

    //  </editor-fold>
    //---------------------------Public Methods--------------------------------- 
    // <editor-fold defaultstate="collapsed" desc="Public Methods">
    public void addIntron(Intron intron) {
        if (introns == null) {
            introns = new ArrayList();
        }
        introns.add(intron);
    }

    //---------------------------------------
    public void addExon(Exon exon) {
        if (exons == null) {
            exons = new ArrayList();
        }
        exons.add(exon);
    }

    //---------------------------------------
    /**
     * Metodo para INFERIR los exones de un gen a partir de sus intrones ya
     * definidos donde la lista de informacion interna que recibe por parametros
     * es usada para definir la informacion que posee cada exon
     */
    public boolean inferExons(List<Information> innerInfo, boolean isWithStops, boolean shouldCheck) {
        Information startInf = this.start;
        Information endInf;
        boolean exonsFine = true;

        int startPos = this.start.position;
        int endPos;

        for (Intron intron : introns) {
            endPos = intron.start.position - 1;
            endInf = innerInfo.get(endPos);

            this.addExon(new Exon(startInf, endInf, innerInfo.subList(startPos + 1, endPos)));// endPos - 1?

            startPos = intron.end.position + 1;
            startInf = innerInfo.get(startPos);
        }

        endPos = this.end.position;
        endInf = this.end;

        this.addExon(new Exon(startInf, endInf, innerInfo.subList(startPos + 1, endPos)));

        if (shouldCheck) { // Se evalua si los exones son multiplos de tres y solo contienen una parada en cuadratura.
            exonsFine = exonsTripletCheck();
            //exonsFine = this.exonsCheck(isWithStops);
        }

        return (exonsFine);

    }
    
    //---------------------------------------
    /**
     * Metodo para INFERIR los exones de un gen a partir de sus intrones ya
     * definidos donde la lista de informacion interna que recibe por parametros
     * es usada para definir la informacion que posee cada exon
     */
   
    
    public void inferExons(GeneConstructor constructor) {
        
        Information startPos = this.start, endPos;       

        for (Intron intron : introns) {
            endPos = constructor.getData(intron.start.position - 1);
    
            this.addExon(new Exon(startPos, endPos)); // Restar uno, no sumar.
            startPos = constructor.getData(intron.end.position + 1);
            //startInf = innerInfo.get(startPos);
        }

        endPos = this.end;
   //     endInf = this.end;

        this.addExon(new Exon(startPos, endPos));
    }

    public boolean exonsCheck(boolean isWithStops) {

        int exonsLength = exons.size(), exonsCount = 1;
        boolean exonsOK = true;

        for (Exon exon : exons) {

            //System.out.println(exon.toString());

            String exonChecking = exon.toString();

            if (exonsCount < exonsLength) {


                // Si todos los exones excepto el ultimo se revisan y no tienen paradas entonces exonsOK se mantiene true
                // caso contrario pasa a false.

                if ((exonChecking.indexOf("taa", 0) != -1) || (exonChecking.indexOf("tag", 0) != -1) || (exonChecking.indexOf("tga", 0) != -1)) {

                    exonsOK = false;
                    break;

                }


            }/**/

            if (exonsOK && (exonsCount == exonsLength) && isWithStops) { // Si todos los exones excepto el ultimo estan bien, entro a chequear el ultimo.
                //Reviso que el ultimo exon contenga solo una parada final del mismo.
                //Si el exon cumple la restriccion entonces exonsOK se mantiene true de lo contrario pasa a false



                if ((exonChecking.indexOf("taa", 0)) != -1) {

                    int index = exonChecking.indexOf("taa", 0);
                    if (!(index == (exonChecking.length() - 3))) {
                        exonsOK = false;
                        break;
                    }
                }

                if ((exonChecking.indexOf("tga", 0)) != -1) {

                    int index = exonChecking.indexOf("tga", 0);
                    if (!(index == (exonChecking.length() - 3))) {
                        exonsOK = false;
                        break;
                    }
                }

                if ((exonChecking.indexOf("tag", 0)) != -1) {

                    int index = exonChecking.indexOf("tag", 0);
                    if (!(index == (exonChecking.length() - 3))) {
                        exonsOK = false;
                        break;
                    }
                }


            }

            if (exonsOK && (exonsCount == exonsLength) && !isWithStops) {

                if ((exonChecking.indexOf("taa", 0) != -1) || (exonChecking.indexOf("tag", 0) != -1) || (exonChecking.indexOf("tga", 0) != -1)) {

                    exonsOK = false;
                    break;

                }
            }

            exonsCount++;

        }

        return exonsOK;
    }

    public boolean exonsTripletCheck() {


        boolean exonsOK = true; // Se asume que al entrar los exones estan bien.

        String exonChecking = "", exonCheckingT = ""; // contendra todos los exones concatenados.

        for (Exon exon : exons) {

            //System.out.println(exon.toString());

            exonChecking += exon.toString();

        }

        //exonChecking = "atgagaatagaagataaatga";

        String exonCheckingSub = exonChecking; // Contedra el resto de exones concatenados por revisar

        int longitudConcatenados = exonChecking.length();

        int multiplicidad, cuadratura;

        multiplicidad = (longitudConcatenados % 3); // Para determinar si los exones son multiplo de tres.

        if (multiplicidad == 0) {// Si los exones concatenados son multiplos de tres, entonces 
            // se chequea que exista solo una parada en cuadratura y al final de la cadena.

            int posicionParadaSig; // Indica posicion de la primera parada hallada en la subcadena en proceso.

            int posicionParadAnt = 0; // llevara registro de la ultima senal aparente de parada hallada en la cadena global.

            int posicionParadaGlob; // Indica posicion gloal de la parada en proceso.

            int menorIndex; // La posicion de la primera senal de parada aparente hallada en la cadena en proceso.

            List<Integer> listaIndexOf = new ArrayList<>(); // Guarda todas las primera ocurrencias de las senales taa, tga y tag.
            // en la cadena en proceso.

            while (exonsOK) {


                if ((exonCheckingSub.indexOf("taa", 0) != -1) || (exonCheckingSub.indexOf("tag", 0) != -1) || (exonCheckingSub.indexOf("tga", 0) != -1)) {

                    listaIndexOf.clear(); // Al inicio de cada revision de senales borro cualquier senal procesadacon anterioridad.

                    int indexTAA = exonCheckingSub.indexOf("taa", 0);
                    int indexTAG = exonCheckingSub.indexOf("tag", 0);
                    int indexTGA = exonCheckingSub.indexOf("tga", 0);

                    if (indexTAA != -1) {
                        listaIndexOf.add(indexTAA);
                    }
                    if (indexTAG != -1) {
                        listaIndexOf.add(indexTAG);
                    }
                    if (indexTGA != -1) {
                        listaIndexOf.add(indexTGA);
                    }

                    Collections.sort(listaIndexOf);

                    menorIndex = listaIndexOf.get(0);

                    if ((indexTAA != -1) && (indexTAA == menorIndex)) {

                        posicionParadaSig = menorIndex;

                        posicionParadaGlob = posicionParadAnt + posicionParadaSig;

                        cuadratura = posicionParadaGlob % 3;

                        // Si la senal hallada NO es la ultima,
                        if (!(posicionParadaGlob == exonChecking.length() - 3)) {


                            // entonces chequeo cuadratura y si la senal NO esta en cuadratura, avanzo.
                            if (cuadratura != 0) {

                                // exonCheckingSub contendra el resto de exones concatenados para buscar siguiente posible parada.
                                exonCheckingSub = exonCheckingSub.substring((posicionParadaSig + 1), exonCheckingSub.length());
                                posicionParadAnt += (posicionParadaSig + 1);
                                continue; // el while avanza.

                            } else {

                                exonsOK = false;
                                break; // salgo del while, los exones concatenados no son validos.
                            }

                        } else { // Si la parada global coincide con el fin de la cadena, entonces los exones estan bien.

                            break;

                        }


                    }

                    if ((indexTAG != -1) && (indexTAG == menorIndex)) {

                        posicionParadaSig = menorIndex;

                        posicionParadaGlob = posicionParadAnt + posicionParadaSig;

                        cuadratura = posicionParadaGlob % 3;

                        // Si la senal hallada NO es la ultima,
                        if (!(posicionParadaGlob == exonChecking.length() - 3)) {


                            // entonces chequeo cuadratura.
                            if (cuadratura != 0) {

                                // exonCheckingSub contendra el resto de concatenados para buscar siguiente posible parada.
                                exonCheckingSub = exonCheckingSub.substring((posicionParadaSig + 1), exonCheckingSub.length());
                                posicionParadAnt += posicionParadaSig + 1;
                                continue;

                            } else {

                                exonsOK = false;
                                break; // salgo del while, los exones concatenados no son validos.
                            }

                        } else { // Si la parada global coincide con el fin de la cadena, entonces los exones estan bien.

                            break;


                        }


                    }

                    if ((indexTGA != -1) && (indexTGA == menorIndex)) {

                        posicionParadaSig = menorIndex;

                        posicionParadaGlob = posicionParadAnt + posicionParadaSig;

                        cuadratura = posicionParadaGlob % 3;

                        // Si la senal hallada NO es la ultima,
                        if (!(posicionParadaGlob == exonChecking.length() - 3)) {


                            // entonces chequeo cuadratura.
                            if (cuadratura != 0) {

                                // exonCheckingSub contendra el resto de concatenados para buscar siguiente posible parada.
                                exonCheckingSub = exonCheckingSub.substring((posicionParadaSig + 1), exonCheckingSub.length());
                                posicionParadAnt += posicionParadaSig + 1;
                                continue;

                            } else {

                                exonsOK = false;
                                break; // salgo del while, los exones concatenados no son validos.
                            }

                        } else { // Si la parada global coincide con el fin de la cadena, entonces los exones estan bien.

                            break;


                        }


                    }

                }

            }

        } else { // Los exones concatenados no son multiplo de tres, la lectura no es valida y no debe agregarse como CDS.

            exonsOK = false;
        }

        //System.out.println(exonChecking);
        return exonsOK;
    }
//  </editor-fold>
//---------------------------Override Methods------------------------------- 
// <editor-fold defaultstate="collapsed" desc="Override Methods">

    /**
     * Implementacion del metodo de GenePart para hacer el formato de impresion
     * de data para los intrones o exones, recibe un boolean como true para
     * imprimir los intrones y false para los exones
     * <br/><br/>
     * Ejemplo salida:<br/>
     * Formato para Intron = [atg,(gtacgttgcag),(gtgcattcag),taa]<br/>
     * Formato para Exon = [(atgcgactcaaga),(tgcaagtac),(gactatgacataa)]<br/>
     */
    @Override
    public String getStringInfo(boolean intronFormat) {
        String out = "(";

        if (intronFormat) {

            out += start.toString();
            out += this.exons.get(0).getFirstToGene();

            for (Intron intron : introns) {
                out += "," + intron.getStringInfo(intronFormat);
            }

            out += ",";

            if (checkEdges) {
                out += this.exons.get(exons.size() - 1).getLastToGene();
            }

            out += this.end.toString();
        } else {
            for (Exon exon : exons) {
                out += "," + exon.getStringInfo(intronFormat);
            }
            out = out.replaceFirst(",", "");
        }

        out += ")";

        return out;
    }

    //---------------------------------------
    /**
     * Implementacion del metodo de GenePart para hacer el formato de impresion
     * de PARES para los intrones o exones, recibe un boolean como true para
     * imprimir los intrones y false para los exones
     * <br/><br/>
     * Ejemplo salida:<br/>
     * Formato para Intron = [1,(30,100),(130,210),250]<br/>
     * Formato para Exon = [(1,29),(101,129),(211,250)]<br/>
     */
    @Override
    public String getPositionsInfo(boolean intronFormat) {
        String out = "";

        if (intronFormat) {
            out += start.position.toString();

            for (Intron intron : introns) {
                out += "," + intron.getPositionsInfo(intronFormat);
            }

            out += "," + ((checkEdges) ? this.getEnd() : this.end).position.toString();
        } else {
            for (Exon exon : exons) {
                out += "," + exon.getPositionsInfo(intronFormat);
            }
            out = out.replaceFirst(",", "");
        }

        out += "";

        return out;
    }

    //---------------------------------------
    @Override
    public Information getEnd() {
        Information posF = this.exons.get(exons.size() - 1).getEnd();
        
        return posF;
    }

    //---------------------------------------
    @Override
    public String toString() {
        String out = "";
        int exonsSize = exons.size();
        int intronsSize = 0;
        if (introns != null) {
            intronsSize = introns.size();
        }

        for (int i = 0; i < exonsSize; i++) {
            out += exons.get(i).toString();
            //int exonl = exons.get(i).toString().length();

            if (intronsSize != 0) {
                if (i < intronsSize) {
                    out += introns.get(i).toString();
                }
            }
        }

        return out;
    }

    //---------------------------------------
    @Override
    public List<Information> getData() {
        ArrayList<Information> data = new ArrayList<>();

        int exonsSize = exons.size();
        int intronsSize = introns.size();

        for (int i = 0; i < exonsSize; i++) {
            data.addAll(this.exons.get(i).getData());
            if (i < intronsSize) {
                data.addAll(this.introns.get(i).getData());
            }
        }

        return data;
    }
    //  </editor-fold>
}
