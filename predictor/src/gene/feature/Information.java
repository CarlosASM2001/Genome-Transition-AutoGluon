/*
    Information.java
    Clase descriptora de la informacion en cada lugar de un gen, usada solo para
    asociar una data a una posicion del gen y para validar que sea permitida
 

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

import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Clase descriptora de la informacion en cada lugar de un gen, usada solo para
 * asociar una data a una posicion del gen y para validar que sea permitida
 */
public class Information {
    //---------------------------Static Constants-------------------------------
    // <editor-fold desc="Static Constants">

    public static final Pattern lexicalMatcher = Pattern.compile("[^" + Model.allowCharacters + "]+");
    //  </editor-fold>
    //---------------------------Public Attributes------------------------------
    // <editor-fold desc="Public Attributes">
    public Integer position;
    //---------------------------------------
    public String info;
    //  </editor-fold>
    //---------------------------Constructors-----------------------------------
    // <editor-fold defaultstate="collapsed" desc="Constructors">

    public Information(Integer position, String info) throws Exception {
        this.position = position;
        this.info = info;
        Matcher matcher = lexicalMatcher.matcher(info);
        if (matcher.find()) {
            throw new Exception("El caracter [" + position + ":" + matcher.group() + "] "
                    + "no corresponde a un estructura valida de gen");
        }
    }
    //---------------------------------------
    //  </editor-fold>
    //---------------------------Override Methods------------------------------- 
    // <editor-fold defaultstate="collapsed" desc="Override Methods">
    @Override
    public String toString() {
        return info;
    }
    
    //---------------------------------------
    @Override
    public boolean equals(Object obj) {
        if(obj instanceof Information){
            Information other = ((Information) obj);
            return this.position.equals(other.position) && this.info.equals(other.info);
        }
        
        return false;
    }
    
    //---------------------------------------
    @Override
    public int hashCode() {
        int hash = 7;
        hash = 61 * hash + Objects.hashCode(this.position);
        hash = 61 * hash + Objects.hashCode(this.info);
        return hash;
    }
    
    //  </editor-fold>


}
