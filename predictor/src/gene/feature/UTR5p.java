/*
    UTR5p.java
    Modela regiones UTR 5'

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

/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package gene.feature;

import java.util.List;
import java.util.regex.Pattern;

/**
 *
 * @author jose
 */
public class UTR5p extends GenePart {
    
      
    //---------------------------Static Constants-------------------------------
    // <editor-fold desc="Static Constants">
        
    //  </editor-fold>
    //---------------------------Private Attributes-----------------------------
    // <editor-fold desc="Private Attributes">
    private int endStart = 0;
    
    //  </editor-fold>
    //---------------------------Constructors-----------------------------------
    // <editor-fold defaultstate="collapsed" desc="Constructors">

    public UTR5p(Information start, Information end, List<Information> innerInfo) throws Exception {
        super(start, end, innerInfo);

        endStart = innerInfo.size() - 1;
        
    }

    //---------------------------------------
    //  </editor-fold>
    //---------------------------Override Methods------------------------------- 
    // <editor-fold defaultstate="collapsed" desc="Override Methods">
    @Override
    public String getStringInfo(boolean intron) {
        String out = "[";
        out += this.toString();
        out += "]";
        return out;
    }

    //---------------------------------------
    @Override
    public String getPositionsInfo(boolean intron) {
        String out = "[";

        out += start.position.toString() + ",";
        out += getEnd().position.toString();

        out += "]";
        return out;
    }

    //---------------------------------------
    @Override
    public Information getEnd() {
        return this.innerInfo.get(endStart);
    }
    
    //  </editor-fold>
    
}
