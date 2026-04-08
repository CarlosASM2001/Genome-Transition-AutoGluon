/*
    GenePart.java
    Clase abstracta que describe la forma de cualquier parte del gen, metodos
    abstractos solo para personalizar el formato de impresion.

    Copyright (C) 2016.
    Morales Yonathan (yonathan.morales@unet.edu.ve).

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

import java.util.ArrayList;
import java.util.List;

/**
 * Clase abstracta que describe la forma de cualquier parte del gen, metodos
 * abstractos solo para personalizar el formato de impresion
 */
public abstract class GenePart implements Model {
    //---------------------------Protected Attributes---------------------------
    // <editor-fold desc="Protected Attributes">

    protected Information start;
    protected Information end;
    //---------------------------------------
    protected List<Information> innerInfo;

    //  </editor-fold>
    //---------------------------Constructors-----------------------------------
    // <editor-fold defaultstate="collapsed" desc="Constructors">
    public GenePart(Information start, Information end, List<Information> innerInfo) {
        this.start = start;
        this.end = end;
        this.innerInfo = innerInfo;
    }

    //---------------------------------------
    public GenePart(Information start, Information end) {
        this.start = start;
        this.end = end;
    }
    //  </editor-fold>
    //---------------------------Getters---------------------------------------- 
    // <editor-fold defaultstate="collapsed" desc="Getters">

    public Information getStart() {
        return start;
    }

    //---------------------------------------
    public Information getEnd() {
        return end;
    }

    //---------------------------------------
    public List<Information> getInnerInfo() {
        return innerInfo;
    }

    //  </editor-fold>
    //---------------------------Override Methods------------------------------- 
    // <editor-fold defaultstate="collapsed" desc="Override Methods">
    @Override
    public String toString() {
        String out = start.toString();
        for (Information information : innerInfo) {
            out += information.toString();
        }
        out += end.toString();
        return out;
    }
    //---------------------------------------
    //  </editor-fold>
    //---------------------------Public Methods--------------------------------- 
    // <editor-fold defaultstate="collapsed" desc="Public Methods">

    /**
     * Personaliza el modo de impresion de las posiciones de inicio y fin
     * correspondientes a cada parte del gen
     */
    public List<Information> getData() {
        ArrayList<Information> data = new ArrayList<>();
        data.add(this.start);
        
        for (Information information : this.innerInfo) {
            data.add(information);
        }
        
        data.add(this.end);
        return data;
    }
    //  </editor-fold>
    //---------------------------Abstract Methods------------------------------- 
    // <editor-fold defaultstate="collapsed" desc="Abstract Methods">

    /**
     * Personaliza el modo de impresion de la data de una parte del gen
     */
    public abstract String getStringInfo(boolean intron);
    //---------------------------------------

    /**
     * Personaliza el modo de impresion de las posiciones de inicio y fin
     * correspondientes a cada parte del gen
     */
    public abstract String getPositionsInfo(boolean intron);
    //  </editor-fold>
}
