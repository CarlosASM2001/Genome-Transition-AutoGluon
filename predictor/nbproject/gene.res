package gene.feature;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

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
    private List<Intron> introns;
    private List<Exon> exons;
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
    public void inferExons(List<Information> innerInfo) {
        Information startInf = this.start;
        Information endInf;

        int startPos = this.start.position;
        int endPos;

        for (Intron intron : introns) {
            endPos = intron.start.position - 1;
            endInf = innerInfo.get(endPos);

            this.addExon(new Exon(startInf, endInf, innerInfo.subList(startPos + 1, endPos)));

            startPos = intron.end.position + 1;
            startInf = innerInfo.get(startPos);
        }

        endPos = this.end.position;
        endInf = this.end;

        this.addExon(new Exon(startInf, endInf, innerInfo.subList(startPos + 1, endPos)));
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
        return this.exons.get(exons.size() - 1).getLastInfToGene();
    }

    //---------------------------------------
    @Override
    public String toString() {
        String out = "";
        int exonsSize = exons.size();
        int intronsSize = introns.size();

        for (int i = 0; i < exonsSize; i++) {
            out += exons.get(i).toString();
            if (i < intronsSize) {
                out += introns.get(i).toString();
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


public String lectsToString {
        String out = constructor.toString();

        int i = 0;
        for (Gene gene : lectures) {
            out += "\n----------------------------Lectura #" + (++i) + "----------------------------------\n";
            out += "GEN = " + gene.toString() + "\n";
            out += "----------------------------------------------------\n";
            //out += "N I N  " + gene.getPositionsInfo(true) + "\n";
            out += "Intrones DATA = " + gene.getStringInfo(true) + "\n";
            out += "----------------------------------------------------\n";
            out += "N E N  " + gene.getPositionsInfo(false).replace("(", "[").replace(")", "]") + "\n";
            out += "Exones DATA = " + gene.getStringInfo(false) + "\n";
        }

        return out;
    }
