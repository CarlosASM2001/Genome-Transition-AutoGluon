 /*
    Exon.java
    Clase que extiende de GenePart, como implementacion especifica de que es un
    Exon, solo implementa sus metodos de escritura para permitir el formato en
    modo exon tanto para su DATA como para sus PARES (posiciones inicio y fin).

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

import java.util.List;

/**
 * Clase que extiende de GenePart, como implementacion especifica de que es un
 * Exon, solo implementa sus metodos de escritura para permitir el formato en
 * modo exon tanto para su DATA como para sus PARES (posiciones inicio y fin).
 */
public class Exon extends GenePart {
    //---------------------------Constructors-----------------------------------
    // <editor-fold defaultstate="collapsed" desc="Constructors">

    public Exon(Information start, Information end, List<Information> innerInfo) {
        super(start, end, innerInfo);
    }
    
    public Exon(Information start, Information end) {
        super(start, end);
    }

    //---------------------------------------
    //  </editor-fold>
    //---------------------------Package Methods-------------------------------- 
    // <editor-fold defaultstate="collapsed" desc="Package Methods">
    String getFirstToGene() {
        String out = this.innerInfo.get(0).toString();
        out += this.innerInfo.get(1).toString();
        return out;
    }
    
    //---------------------------------------
    String getLastToGene() {
        int last = this.innerInfo.size() - 2;
        String out = this.innerInfo.get(last++).toString();
        out += this.innerInfo.get(last).toString();
        return out;
    }
    
    //---------------------------------------
    Information getLastInfToGene(){
        int last = this.innerInfo.size() - 1;
        return this.innerInfo.get(last);
    }
    
    
    //  </editor-fold>
    //---------------------------Override Methods------------------------------- 
    // <editor-fold defaultstate="collapsed" desc="Override Methods">

    @Override
    public String getStringInfo(boolean intron) {
        String out = "(";
        out += this.toString();
        out += ")";
        return out;
    }

    //---------------------------------------
    @Override
    public String getPositionsInfo(boolean intron) {
        String out = "(";

        out += start.position.toString() + ",";
        out += end.position.toString();

        out += ")";
        return out;
    }
    
    //---------------------------------------
    
    //  </editor-fold>
}
