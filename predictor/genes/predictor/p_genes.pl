
% En general se debe consultar:

% gen(G), p_gener(G, Patgs, Pgt, Pag, Ppar, Pint, Gen, R). 

% gen con atg, splicings, parada taa. (funciona)
% p_gen([a,c,a,t,g,t,a,c,t,a,g,g,t,a,c,t,g,t,a,a,a,c,c,c,c,c,a,g,g,t,c,g,c,g,c,c,c,c,c,t,a,a,g,g,t,c,g,c,g], I, PI, P, R).

% gen sin atg, splicings, parada taa. (funciona)
% p_gen([a,g,a,t,t,t,a,c,t,a,g,g,t,a,c,t,g,t,a,a,a,c,c,c,c,c,a,g,g,t,c,g,c,g,c,c,c,c,c,t,a,a,g,g,t,c,g,c,g], I, PI, P, R).

% gen sin atg, splicings, sin parada (funciona)
% p_gen([a,c,a,g,g,g,t,a,c,c,g,t,a,a,a,c,c,c,a,g,a,g,g,g,c,g,c,g,c,c,c,c,c,a,a,a,g,a,t,a,a,c,g,c,g], I, PI, P, R).

% p_gen([c,c,c,c,a,a,a,g,a,t,a,a,c,g,c,g], I, PI, P, R).

% p_gen([a,t,a,a,c,g,c,g], I, PI, P, R).

% p_gen([a,c,a,t,g,c,t,a,g,g,g,t,a,a,a,c,c,c,a,g,a,g,g,g,c,g,c,g,c,c,c,c,c,a,a,a,g,a,t,a,a,c,g,c,g], I, PI, P, R).

% senal_atg([a,g,a,t,g,t,a,c,a,g,c,g,t,a,c,c,g,t,a,a,a,c,c,c,a,g,a,g,g,g,c,g,c,g,c,c,c,c,c,a,a,a,g,a,t,a,a,c,g,c,g], 'atg', Patg, RSecuencia1).

% senal_gt([a,c,a,g,g,g,t,a,c,c,g,t,a,a,a,c,c,c,a,g,a,g,g,g,c,g,c,g,c,c,c,c,c,a,a,a,g,a,t,a,a,c,g,c,g], 'gt', P, RSecuencia1).

% senal_ag([a,c,a,t,g,g,t,a,c,c,g,t,a,a,a,c,c,c,a,g,a,g,g,g,c,g,c,g,c,c,c,c,c,a,a,a,g,a,t,a,a,c,g,c,g], 'ag', P, RSecuencia1).

% senal_par([a,g,a,t,g,t,a,c,a,g,c,g,t,a,c,c,g,t,a,a,a,c,c,c,a,g,a,g,g,g,c,g,c,g,c,c,c,c,c,a,a,a,g,a,t,a,a,c,g,c,g], 'taa', P, RSecuencia1).

% conservados_gt([a,c,a,t,g,c,t,a,g,g,g,t,a,a,a,c,c,c,a,g,a,g,g,g,c,g,c,g,c,c,c,c,c,a,a,a,g,a,t,a,a,c,g,c,g], 'gt', 'ambas', Posicion, RSecuencia).

% conservados_ag([a,c,a,t,g,c,t,c,g,g,g,t,a,a,a,c,c,c,a,g,a,g,g,g,c,g,c,g,c,c,c,c,c,a,a,a,g,a,t,a,a,c,g,c,g], 'ag', 'der', Posicion, RSecuencia).

% conservados_atg([a,g,a,t,g,t,a,c,a,g,c,g,t,a,c,c,g,t,a,a,a,c,c,c,a,g,a,g,g,g,c,g,c,g,c,c,c,c,c,a,a,a,g,a,t,a,a,c,g,c,g], 'atg', 'izq', Posicion, RSecuencia).

% conservados_par([a,c,a,a,t,g,t,a,c,c,g,g,a,a,a,c,c,c,a,g,g,c,a,g,c,g,c,g,c,c,c,c,c,a,a,c,t,a,t,a,a,c,g,c,g], 'taa', 'ambas', Posicion, RSecuencia).

% p_intrones([a,c,a,t,g,g,t,a,c,c,g,t,a,a,a,c,c,c,a,g,a,g,g,g,c,g,c,g,c,c,c,c,c,a,a,a,g,a,t,a,a,c,g,c,g], Posiciones, RSecuencia, 0).

% trans_AT([g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,a,c,a,t,g,g,t,g,t,a,g,t,a,g,g,g,t,a,a,a,c,c,c,a,g,g,g,g,g,g,t,a,a,a,c,c,c,a,g,g,g,a,g,g,g,t,a,a,a,c,c,c,a,g,g,g,t,a,a,a,c,c,c,a,g,a,g,g,g,c,g,c,g,c,c,c,c,c,a,a,a,g,a,t,a,a,c,g,c,g], Posicion, 'atg', 'cod->no_cod', ZD).

% trans_AT([a,t,g,a,t,a,a,a,a,t,t,a,a,a,a,a,a,t,t,t,t,t,t,a,a,a,t,a,t,t,t,a,t,a,a,t,t,t,a,a,a,t,t,a,t,a,a,a,t,a,t,c,a,a,t,a,t,a,a,a,a,a,t,a,a,a,a,t,a,a,a,t,t,t,a,t,a,t,t,t,a,a,t,a,a,g,a,c,a,a,g,g,a,t,t,a,a,a,t,a,t,a,a,a,t,t,t,a,a,t,a,a,a,a,a,a,t,t,t,a,t,c,t,a,g,t,a,a,t,a,t,t,t,t,t,t,t,a,t,a,t,a,t,g,t,t,t,a,t,t,t,a,t,a,a,t,t,t,t,a,a,a,a,a,a,t,a,t,t,c,t,t,t,a,a,a,a,t,t,a,t,t,a,a,a,t,a,t,a,t,t,t,a,a,a,t,t,a,c,c,t,g,a,t,t,g,g,a,a,t,t,t,t,t,t,t,g,a,t,t,g,t,c,c,a,a,a,t,a,t,a,a,a,t,t,a,t,g,a,t,a,a,t,a,t,t,a,t,t,t,a,t,t,a,t,t,c,t,t,c,t,a,t,t,t,t,a,a,a,a,g,a,t,a,a,t,a,a,t,t,t,a,a,t,a,t,a,t,t,a,t,t,t,a,a,a,a,a,a,t,a,a,t,t,t,a,a,a,t,a,t,t,g,a,a,t,t,t,t,t,a,g,a,t,a,g,t,a,t,a,t,t,a,a,t,a,a,a,a,a,a,t,a,a,t,t,c,t,a,t,a,g,a,t,a,t,t,a,t,a,t,t,t,g,a,t,a,g,t,a,t,g,t,c,t,a,t,t,t,t,a,c,a,t,a,c,t,a,c,a,c,a,a,t,a,t,t,t,t,t,t,a,a,a,a,a,a,a,t,t,a,g,g,a,a,t,a,a,t,t,t,t,t,t,t,a,c,c,t,t,t,a,t,t,t,g,a,t,a,t,t,a,t,a,t,t,t,a,a,a,t,a,t,c,c,t,t,t,a,t,t,a,a,t,a,a,a,a,a,a,a,t,a,t,t,t,a,g,g,t,a,c,t,a,t,t,a,t,t,t,c,t,t,a,t,a,a,a,g,a,t,a,a,t,t,t,t,t,t,t,g,c,t,a,a,t,a,t,t,a,a,t,t,c,a,a,t,a,a,t,a,t,t,t,a,g,t,g,a,a,g,g,a,t,c,t,t,t,t,t,g,t,t,a,t,a,t,a,c,c,t,a,a,a,t,a,t,g,t,a,a,a,g,t,g,t,a,a,t,t,t,t,a,a,t,t,t,a,t,c,a,a,c,a,t,a,t,t,t,t,a,a,a,a,c,t,a,a,t,t,c,t,t,c,t,g,a,t,t,t,t,g,c,a,c,a,a,t,t,t,g,a,a,c,g,t,a,c,t,t,t,a,a,t,a,a,t,a,g,t,t,g,g,t,a,a,a,t,a,t,t,c,t,t,a,t,g,t,a,t,c,a,t,a,t,t,t,a,g,a,a,g,g,a,t,g,t,a,c,a,g,c,t,t,c,a,t,t,a,t,a,t,a,a,a,g,a,a,t,c,a,c,a,a,t,t,a,c,a,t,g,t,a,g,c,t,a,t,a,g,t,a,g,a,a,a,t,a,a,t,a,g,t,a,a,a,a,g,a,t,t,a,t,g,g,t,t,a,t,a,t,a,a,a,a,t,a,t,t,a,t,a,c,a,t,t,a,c,a,a,a,a,t,t,g,g,t,a,t,a,g,a,g,g,a,g,a,t,t,a,t,t,t,a,g,g,t,a,a,t,g,g,t,g,g,t,t,t,a,t,a,t,a,a,t,t,t,t,a,c,a,a,c,t,a,a,a,c,g,t,g,g,t,a,t,a,t,g,t,t,t,a,a,a,t,t,a,t,t,c,a,a,a,a,t,t,a,g,a,t,t,g,g,a,t,a,c,a,a,g,t,t,g,a,a,g,t,a,g,g,t,t,c,g,a,t,t,a,t,a,a,c,a,t,g,g,a,a,a,t,a,c,c,c,t,t,c,t,a,c,t,a,t,t,t,t,a,a,a,a,g,g,t,a,a,a,t,t,t,t,c,t,a,t,t,a,g,t,a,a,t,t,t,t,t,a,t,t,c,a,a,t,a,t,c,t,t,t,t,a,t,a,t,c,a,a,a,t,a,t,g,c,a,a,a,t,a,g,c,t,g,a,t,a,c,t,g,g,t,a,g,t,a,a,a,a,t,g,t,a,t,c,a,t,a,t,a,g,g,a,t,c,t,t,a,t,a,c,t,a,a,a,a,g,t,t,a,t,a,t,a,a,t,t,t,c,t,a,a,a,a,g,t,a,t,a,t,c,t,t,t,a,a,a,t,a,a,c,t,c,a,t,t,a,a,a,t,a,t,a,t,t,t,a,g,a,g,g,t,t,t,a,g,t,a,t,a,t,a,t,t,a,a,a,c,c,t,t,t,t,t,c,a,t,a,t,a,a,a,t,c,t,t,a,t,a,a,t,t,a,t,a,c,t,g,a,a,t,g,t,a,g,t,t,c,t,t,t,a,a,t,a,t,t,t,g,g,t,a,a,t,a,a,t,t,c,t,t,t,a,a,c,a,g,t,a,a,c,t,a,t,t,c,c,t,t,a,t,a,t,a,a,a,a,a,a,t,t,a,t,a,a,t,a,a,t,a,c,t,a,g,t,t,a,t,g,t,a,a,a,a,c,a,a,g,a,a,g,c,t,t,t,t,g,t,t,t,c,t,a,a,a,a,t,t,g,a,a,a,t,t,a,t,a,t,a,t,t,t,a,t,t,t,t,t,a,t,t,a,a,t,g,c,a,a,c,g,t,g,g,t,t,t,a,a,g,t,a,t,t,t,c,a,g,a,g,t,c,t,a,t,t,t,c,a,t,t,a,t,t,a,a,t,t,a,t,a,g,g,t,t,t,t,t,g,t,t,c,t,g,a,t,a,t,t,t,a,t,a,a,t,a,a,a,t,t,a,c,c,g,t,t,t,g,a,a,t,t,t,a,a,t,t,t,a,g,a,g,a,t,a,c,c,t,a,t,a,t,t,a,t,t,t,t,c,a,t,t,a,a,a,a,a,t,t,a,a,a,g,a,t,a,t,a,t,t,t,a,a,t,t,a,a], Posicion, 'gt', 'cod->no_cod', ZD).

% conservados_gt([a,t,g,a,t,a,a,a,a,t,t,a,a,a,a,a,a,t,t,t,t,t,t,a,a,a,t,a,t,t,t,a,t,a,a,t,t,t,a,a,a,t,t,a,t,a,a,a,t,a,t,c,a,a,t,a,t,a,a,a,a,a,t,a,a,a,a,t,a,a,a,t,t,t,a,t,a,t,t,t,a,a,t,a,a,g,a,c,a,a,g,g,a,t,t,a,a,a,t,a,t,a,a,a,t,t,t,a,a,t,a,a,a,a,a,a,t,t,t,a,t,c,t,a,g,t,a,a,t,a,t,t,t,t,t,t,t,a,t,a,t,a,t,g,t,t,t,a,t,t,t,a,t,a,a,t,t,t,t,a,a,a,a,a,a,t,a,t,t,c,t,t,t,a,a,a,a,t,t,a,t,t,a,a,a,t,a,t,a,t,t,t,a,a,a,t,t,a,c,c,t,g,a,t,t,g,g,a,a,t,t,t,t,t,t,t,g,a,t,t,g,t,c,c,a,a,a,t,a,t,a,a,a,t,t,a,t,g,a,t,a,a,t,a,t,t,a,t,t,t,a,t,t,a,t,t,c,t,t,c,t,a,t,t,t,t,a,a,a,a,g,a,t,a,a,t,a,a,t,t,t,a,a,t,a,t,a,t,t,a,t,t,t,a,a,a,a,a,a,t,a,a,t,t,t,a,a,a,t,a,t,t,g,a,a,t,t,t,t,t,a,g,a,t,a,g,t,a,t,a,t,t,a,a,t,a,a,a,a,a,a,t,a,a,t,t,c,t,a,t,a,g,a,t,a,t,t,a,t,a,t,t,t,g,a,t,a,g,t,a,t,g,t,c,t,a,t,t,t,t,a,c,a,t,a,c,t,a,c,a,c,a,a,t,a,t,t,t,t,t,t,a,a,a,a,a,a,a,t,t,a,g,g,a,a,t,a,a,t,t,t,t,t,t,t,a,c,c,t,t,t,a,t,t,t,g,a,t,a,t,t,a,t,a,t,t,t,a,a,a,t,a,t,c,c,t,t,t,a,t,t,a,a,t,a,a,a,a,a,a,a,t,a,t,t,t,a,g,g,t,a,c,t,a,t,t,a,t,t,t,c,t,t,a,t,a,a,a,g,a,t,a,a,t,t,t,t,t,t,t,g,c,t,a,a,t,a,t,t,a,a,t,t,c,a,a,t,a,a,t,a,t,t,t,a,g,t,g,a,a,g,g,a,t,c,t,t,t,t,t,g,t,t,a,t,a,t,a,c,c,t,a,a,a,t,a,t,g,t,a,a,a,g,t,g,t,a,a,t,t,t,t,a,a,t,t,t,a,t,c,a,a,c,a,t,a,t,t,t,t,a,a,a,a,c,t,a,a,t,t,c,t,t,c,t,g,a,t,t,t,t,g,c,a,c,a,a,t,t,t,g,a,a,c,g,t,a,c,t,t,t,a,a,t,a,a,t,a,g,t,t,g,g,t,a,a,a,t,a,t,t,c,t,t,a,t,g,t,a,t,c,a,t,a,t,t,t,a,g,a,a,g,g,a,t,g,t,a,c,a,g,c,t,t,c,a,t,t,a,t,a,t,a,a,a,g,a,a,t,c,a,c,a,a,t,t,a,c,a,t,g,t,a,g,c,t,a,t,a,g,t,a,g,a,a,a,t,a,a,t,a,g,t,a,a,a,a,g,a,t,t,a,t,g,g,t,t,a,t,a,t,a,a,a,a,t,a,t,t,a,t,a,c,a,t,t,a,c,a,a,a,a,t,t,g,g,t,a,t,a,g,a,g,g,a,g,a,t,t,a,t,t,t,a,g,g,t,a,a,t,g,g,t,g,g,t,t,t,a,t,a,t,a,a,t,t,t,t,a,c,a,a,c,t,a,a,a,c,g,t,g,g,t,a,t,a,t,g,t,t,t,a,a,a,t,t,a,t,t,c,a,a,a,a,t,t,a,g,a,t,t,g,g,a,t,a,c,a,a,g,t,t,g,a,a,g,t,a,g,g,t,t,c,g,a,t,t,a,t,a,a,c,a,t,g,g,a,a,a,t,a,c,c,c,t,t,c,t,a,c,t,a,t,t,t,t,a,a,a,a,g,g,t,a,a,a,t,t,t,t,c,t,a,t,t,a,g,t,a,a,t,t,t,t,t,a,t,t,c,a,a,t,a,t,c,t,t,t,t,a,t,a,t,c,a,a,a,t,a,t,g,c,a,a,a,t,a,g,c,t,g,a,t,a,c,t,g,g,t,a,g,t,a,a,a,a,t,g,t,a,t,c,a,t,a,t,a,g,g,a,t,c,t,t,a,t,a,c,t,a,a,a,a,g,t,t,a,t,a,t,a,a,t,t,t,c,t,a,a,a,a,g,t,a,t,a,t,c,t,t,t,a,a,a,t,a,a,c,t,c,a,t,t,a,a,a,t,a,t,a,t,t,t,a,g,a,g,g,t,t,t,a,g,t,a,t,a,t,a,t,t,a,a,a,c,c,t,t,t,t,t,c,a,t,a,t,a,a,a,t,c,t,t,a,t,a,a,t,t,a,t,a,c,t,g,a,a,t,g,t,a,g,t,t,c,t,t,t,a,a,t,a,t,t,t,g,g,t,a,a,t,a,a,t,t,c,t,t,t,a,a,c,a,g,t,a,a,c,t,a,t,t,c,c,t,t,a,t,a,t,a,a,a,a,a,a,t,t,a,t,a,a,t,a,a,t,a,c,t,a,g,t,t,a,t,g,t,a,a,a,a,c,a,a,g,a,a,g,c,t,t,t,t,g,t,t,t,c,t,a,a,a,a,t,t,g,a,a,a,t,t,a,t,a,t,a,t,t,t,a,t,t,t,t,t,a,t,t,a,a,t,g,c,a,a,c,g,t,g,g,t,t,t,a,a,g,t,a,t,t,t,c,a,g,a,g,t,c,t,a,t,t,t,c,a,t,t,a,t,t,a,a,t,t,a,t,a,g,g,t,t,t,t,t,g,t,t,c,t,g,a,t,a,t,t,t,a,t,a,a,t,a,a,a,t,t,a,c,c,g,t,t,t,g,a,a,t,t,t,a,a,t,t,t,a,g,a,g,a,t,a,c,c,t,a,t,a,t,t,a,t,t,t,t,c,a,t,t,a,a,a,a,a,t,t,a,a,a,g,a,t,a,t,a,t,t,t,a,a,t,t,a,a], 'gt', 'ambas', Posicion, RSecuencia).

% trans_GC([g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,g,c,a,c,a,t,g,g,t,g,t,a,g,t,a,g,g,g,t,a,a,a,c,c,c,a,g,g,g,g,g,g,t,a,a,a,c,c,c,a,g,g,g,a,g,g,g,t,a,a,a,c,c,c,a,g,g,g,t,a,a,a,c,c,c,a,g,a,g,g,g,c,g,c,g,c,c,c,c,c,a,a,a,g,a,t,a,a,c,g,c,g], Posicion, 'atg', 'cod->no_cod', ZD).

% zona_rica_pirimidinas([a,t,g,a,t,a,a,a,a,t,t,a,a,a,a,a,a,t,t,t,t,t,t,a,a,a,t,a,t,t,t,a,t,a,a,t,t,t,a,a,a,t,t,a,t,a,a,a,t,a,t,c,a,a,t,a,t,a,a,a,a,a,t,a,a,a,a,t,a,a,a,t,t,t,a,t,a,t,t,t,a,a,t,a,a,g,a,c,a,a,g,g,a,t,t,a,a,a,t,a,t,a,a,a,t,t,t,a,a,t,a,a,a,a,a,a,t,t,t,a,t,c,t,a,g,t,a,a,t,a,t,t,t,t,t,t,t,a,t,a,t,a,t,g,t,t,t,a,t,t,t,a,t,a,a,t,t,t,t,a,a,a,a,a,a,t,a,t,t,c,t,t,t,a,a,a,a,t,t,a,t,t,a,a,a,t,a,t,a,t,t,t,a,a,a,t,t,a,c,c,t,g,a,t,t,g,g,a,a,t,t,t,t,t,t,t,g,a,t,t,g,t,c,c,a,a,a,t,a,t,a,a,a,t,t,a,t,g,a,t,a,a,t,a,t,t,a,t,t,t,a,t,t,a,t,t,c,t,t,c,t,a,t,t,t,t,a,a,a,a,g,a,t,a,a,t,a,a,t,t,t,a,a,t,a,t,a,t,t,a,t,t,t,a,a,a,a,a,a,t,a,a,t,t,t,a,a,a,t,a,t,t,g,a,a,t,t,t,t,t,a,g,a,t,a,g,t,a,t,a,t,t,a,a,t,a,a,a,a,a,a,t,a,a,t,t,c,t,a,t,a,g,a,t,a,t,t,a,t,a,t,t,t,g,a,t,a,g,t,a,t,g,t,c,t,a,t,t,t,t,a,c,a,t,a,c,t,a,c,a,c,a,a,t,a,t,t,t,t,t,t,a,a,a,a,a,a,a,t,t,a,g,g,a,a,t,a,a,t,t,t,t,t,t,t,a,c,c,t,t,t,a,t,t,t,g,a,t,a,t,t,a,t,a,t,t,t,a,a,a,t,a,t,c,c,t,t,t,a,t,t,a,a,t,a,a,a,a,a,a,a,t,a,t,t,t,a,g,g,t,a,c,t,a,t,t,a,t,t,t,c,t,t,a,t,a,a,a,g,a,t,a,a,t,t,t,t,t,t,t,g,c,t,a,a,t,a,t,t,a,a,t,t,c,a,a,t,a,a,t,a,t,t,t,a,g,t,g,a,a,g,g,a,t,c,t,t,t,t,t,g,t,t,a,t,a,t,a,c,c,t,a,a,a,t,a,t,g,t,a,a,a,g,t,g,t,a,a,t,t,t,t,a,a,t,t,t,a,t,c,a,a,c,a,t,a,t,t,t,t,a,a,a,a,c,t,a,a,t,t,c,t,t,c,t,g,a,t,t,t,t,g,c,a,c,a,a,t,t,t,g,a,a,c,g,t,a,c,t,t,t,a,a,t,a,a,t,a,g,t,t,g,g,t,a,a,a,t,a,t,t,c,t,t,a,t,g,t,a,t,c,a,t,a,t,t,t,a,g,a,a,g,g,a,t,g,t,a,c,a,g,c,t,t,c,a,t,t,a,t,a,t,a,a,a,g,a,a,t,c,a,c,a,a,t,t,a,c,a,t,g,t,a,g,c,t,a,t,a,g,t,a,g,a,a,a,t,a,a,t,a,g,t,a,a,a,a,g,a,t,t,a,t,g,g,t,t,a,t,a,t,a,a,a,a,t,a,t,t,a,t,a,c,a,t,t,a,c,a,a,a,a,t,t,g,g,t,a,t,a,g,a,g,g,a,g,a,t,t,a,t,t,t,a,g,g,t,a,a,t,g,g,t,g,g,t,t,t,a,t,a,t,a,a,t,t,t,t,a,c,a,a,c,t,a,a,a,c,g,t,g,g,t,a,t,a,t,g,t,t,t,a,a,a,t,t,a,t,t,c,a,a,a,a,t,t,a,g,a,t,t,g,g,a,t,a,c,a,a,g,t,t,g,a,a,g,t,a,g,g,t,t,c,g,a,t,t,a,t,a,a,c,a,t,g,g,a,a,a,t,a,c,c,c,t,t,c,t,a,c,t,a,t,t,t,t,a,a,a,a,g,g,t,a,a,a,t,t,t,t,c,t,a,t,t,a,g,t,a,a,t,t,t,t,t,a,t,t,c,a,a,t,a,t,c,t,t,t,t,a,t,a,t,c,a,a,a,t,a,t,g,c,a,a,a,t,a,g,c,t,g,a,t,a,c,t,g,g,t,a,g,t,a,a,a,a,t,g,t,a,t,c,a,t,a,t,a,g,g,a,t,c,t,t,a,t,a,c,t,a,a,a,a,g,t,t,a,t,a,t,a,a,t,t,t,c,t,a,a,a,a,g,t,a,t,a,t,c,t,t,t,a,a,a,t,a,a,c,t,c,a,t,t,a,a,a,t,a,t,a,t,t,t,a,g,a,g,g,t,t,t,a,g,t,a,t,a,t,a,t,t,a,a,a,c,c,t,t,t,t,t,c,a,t,a,t,a,a,a,t,c,t,t,a,t,a,a,t,t,a,t,a,c,t,g,a,a,t,g,t,a,g,t,t,c,t,t,t,a,a,t,a,t,t,t,g,g,t,a,a,t,a,a,t,t,c,t,t,t,a,a,c,a,g,t,a,a,c,t,a,t,t,c,c,t,t,a,t,a,t,a,a,a,a,a,a,t,t,a,t,a,a,t,a,a,t,a,c,t,a,g,t,t,a,t,g,t,a,a,a,a,c,a,a,g,a,a,g,c,t,t,t,t,g,t,t,t,c,t,a,a,a,a,t,t,g,a,a,a,t,t,a,t,a,t,a,t,t,t,a,t,t,t,t,t,a,t,t,a,a,t,g,c,a,a,c,g,t,g,g,t,t,t,a,a,g,t,a,t,t,t,c,a,g,a,g,t,c,t,a,t,t,t,c,a,t,t,a,t,t,a,a,t,t,a,t,a,g,g,t,t,t,t,t,g,t,t,c,t,g,a,t,a,t,t,t,a,t,a,a,t,a,a,a,t,t,a,c,c,g,t,t,t,g,a,a,t,t,t,a,a,t,t,t,a,g,a,g,a,t,a,c,c,t,a,t,a,t,t,a,t,t,t,t,c,a,t,t,a,a,a,a,a,t,t,a,a,a,g,a,t,a,t,a,t,t,t,a,a,t,t,a,a],'ag', 'izq', P, PosSenal, RSecuencia).

% zona_poliadenil([a,t,g,a,t,a,a,a,a,t,t,a,a,a,a,a,a,t,t,t,t,t,t,a,a,a,t,a,t,t,t,a,t,a,a,t,t,t,a,a,a,t,t,a,t,a,a,a,t,a,t,c,a,a,t,a,t,a,a,a,a,a,t,a,a,a,a,t,a,a,a,t,t,t,a,t,a,t,t,t,a,a,t,a,a,g,a,c,a,a,g,g,a,t,t,a,a,a,t,a,t,a,a,a,t,t,t,a,a,t,a,a,a,a,a,a,t,t,t,a,t,c,t,a,g,t,a,a,t,a,t,t,t,t,t,t,t,a,t,a,t,a,t,g,t,t,t,a,t,t,t,a,t,a,a,t,t,t,t,a,a,a,a,a,a,t,a,t,t,c,t,t,t,a,a,a,a,t,t,a,t,t,a,a,a,t,a,t,a,t,t,t,a,a,a,t,t,a,c,c,t,g,a,t,t,g,g,a,a,t,t,t,t,t,t,t,g,a,t,t,g,t,c,c,a,a,a,t,a,t,a,a,a,t,t,a,t,g,a,t,a,a,t,a,t,t,a,t,t,t,a,t,t,a,t,t,c,t,t,c,t,a,t,t,t,t,a,a,a,a,g,a,t,a,a,t,a,a,t,t,t,a,a,t,a,t,a,t,t,a,t,t,t,a,a,a,a,a,a,t,a,a,t,t,t,a,a,a,t,a,t,t,g,a,a,t,t,t,t,t,a,g,a,t,a,g,t,a,t,a,t,t,a,a,t,a,a,a,a,a,a,t,a,a,t,t,c,t,a,t,a,g,a,t,a,t,t,a,t,a,t,t,t,g,a,t,a,g,t,a,t,g,t,c,t,a,t,t,t,t,a,c,a,t,a,c,t,a,c,a,c,a,a,t,a,t,t,t,t,t,t,a,a,a,a,a,a,a,t,t,a,g,g,a,a,t,a,a,t,t,t,t,t,t,t,a,c,c,t,t,t,a,t,t,t,g,a,t,a,t,t,a,t,a,t,t,t,a,a,a,t,a,t,c,c,t,t,t,a,t,t,a,a,t,a,a,a,a,a,a,a,t,a,t,t,t,a,g,g,t,a,c,t,a,t,t,a,t,t,t,c,t,t,a,t,a,a,a,g,a,t,a,a,t,t,t,t,t,t,t,g,c,t,a,a,t,a,t,t,a,a,t,t,c,a,a,t,a,a,t,a,t,t,t,a,g,t,g,a,a,g,g,a,t,c,t,t,t,t,t,g,t,t,a,t,a,t,a,c,c,t,a,a,a,t,a,t,g,t,a,a,a,g,t,g,t,a,a,t,t,t,t,a,a,t,t,t,a,t,c,a,a,c,a,t,a,t,t,t,t,a,a,a,a,c,t,a,a,t,t,c,t,t,c,t,g,a,t,t,t,t,g,c,a,c,a,a,t,t,t,g,a,a,c,g,t,a,c,t,t,t,a,a,t,a,a,t,a,g,t,t,g,g,t,a,a,a,t,a,t,t,c,t,t,a,t,g,t,a,t,c,a,t,a,t,t,t,a,g,a,a,g,g,a,t,g,t,a,c,a,g,c,t,t,c,a,t,t,a,t,a,t,a,a,a,g,a,a,t,c,a,c,a,a,t,t,a,c,a,t,g,t,a,g,c,t,a,t,a,g,t,a,g,a,a,a,t,a,a,t,a,g,t,a,a,a,a,g,a,t,t,a,t,g,g,t,t,a,t,a,t,a,a,a,a,t,a,t,t,a,t,a,c,a,t,t,a,c,a,a,a,a,t,t,g,g,t,a,t,a,g,a,g,g,a,g,a,t,t,a,t,t,t,a,g,g,t,a,a,t,g,g,t,g,g,t,t,t,a,t,a,t,a,a,t,t,t,t,a,c,a,a,c,t,a,a,a,c,g,t,g,g,t,a,t,a,t,g,t,t,t,a,a,a,t,t,a,t,t,c,a,a,a,a,t,t,a,g,a,t,t,g,g,a,t,a,c,a,a,g,t,t,g,a,a,g,t,a,g,g,t,t,c,g,a,t,t,a,t,a,a,c,a,t,g,g,a,a,a,t,a,c,c,c,t,t,c,t,a,c,t,a,t,t,t,t,a,a,a,a,g,g,t,a,a,a,t,t,t,t,c,t,a,t,t,a,g,t,a,a,t,t,t,t,t,a,t,t,c,a,a,t,a,t,c,t,t,t,t,a,t,a,t,c,a,a,a,t,a,t,g,c,a,a,a,t,a,g,c,t,g,a,t,a,c,t,g,g,t,a,g,t,a,a,a,a,t,g,t,a,t,c,a,t,a,t,a,g,g,a,t,c,t,t,a,t,a,c,t,a,a,a,a,g,t,t,a,t,a,t,a,a,t,t,t,c,t,a,a,a,a,g,t,a,t,a,t,c,t,t,t,a,a,a,t,a,a,c,t,c,a,t,t,a,a,a,t,a,t,a,t,t,t,a,g,a,g,g,t,t,t,a,g,t,a,t,a,t,a,t,t,a,a,a,c,c,t,t,t,t,t,c,a,t,a,t,a,a,a,t,c,t,t,a,t,a,a,t,t,a,t,a,c,t,g,a,a,t,g,t,a,g,t,t,c,t,t,t,a,a,t,a,t,t,t,g,g,t,a,a,t,a,a,t,t,c,t,t,t,a,a,c,a,g,t,a,a,c,t,a,t,t,c,c,t,t,a,t,a,t,a,a,a,a,a,a,t,t,a,t,a,a,t,a,a,t,a,c,t,a,g,t,t,a,t,g,t,a,a,a,a,c,a,a,g,a,a,g,c,t,t,t,t,g,t,t,t,c,t,a,a,a,a,t,t,g,a,a,a,t,t,a,t,a,t,a,t,t,t,a,t,t,t,t,t,a,t,t,a,a,t,g,c,a,a,c,g,t,g,g,t,t,t,a,a,g,t,a,t,t,t,c,a,g,a,g,t,c,t,a,t,t,t,c,a,t,t,a,t,t,a,a,t,t,a,t,a,g,g,t,t,t,t,t,g,t,t,c,t,g,a,t,a,t,t,t,a,t,a,a,t,a,a,a,t,t,a,c,c,g,t,t,t,g,a,a,t,t,t,a,a,t,t,t,a,g,a,g,a,t,a,c,c,t,a,t,a,t,t,a,t,t,t,t,c,a,t,t,a,a,a,a,a,t,t,a,a,a,g,a,t,a,t,a,t,t,t,a,a,t,t,a,a],'tag','der'). 

% p_atgs([a,c,a,t,g,c,t,a,g,g,g,t,a,a,a,c,c,c,a,g,a,g,g,g,c,g,c,g,c,c,c,c,c,a,a,a,g,a,t,a,a,c,g,c,g,a,c,a,t,g,c,t,a,g,g,g,t,a,a,a,c,c,c,a,g,a,g,g,g,c,g,c,g,c,c,c,c,c,a,a,a,g,a,t,a,a,c,g,c,g], P, RS, 0).

% p_gen([a,c,a,t,g,c,t,a,g,g,g,t,a,a,a,c,c,c,a,g,a,g,g,g,c,g,c,g,c,c,c,c,c,a,a,a,g,a,t,a,a,c,g,c,g], I, PI, P, R).

% p_gen([a,c,a,t,g,c,t], I, PI, P, R).

% p_atgs([a,c,a,t,g,c,t], P, RS, 0).

% p_atgs([a,c,a,t,g,t,a,c,t,a,g,g,t,a,c,t,g,t,a,a,a,c,c,c,c,c,a,g,g,t,c,g,c,g,c,c,c,c,c,t,a,a,g,g,t,c,g,c,g], P, RS, 0).

% p_atgs([a,c,a,t,g,c,t,a,g,g,g,t,a,a,a,c,c,c,a,g,a,g,g,g,c,g,c,g,c,c,c,c,c,a,a,a,g,a,t,a,a,c,g,c,g,a,c,a,t,g,c,t,a,g,g,g,t,a,a,a,c,c,c,a,g,a,g,g,g,c,g,c,g,c,c,c,c,c,a,a,a,g,a,t,a,a,c,g,c,g], P, RS, 0).

% p_atgs([a,t,g,a,t,a,a,a,a,t,t,a,a,a,a,a,a,t,t,t,t,t,t,a,a,a,t,a,t,t,t,a,t,a,a,t,t,t,a,a,a,t,t,a,t,a,a,a,t,a,t,c,a,a,t,a,t,a,a,a,a,a,t,a,a,a,a,t,a,a,a,t,t,t,a,t,a,t,t,t,a,a,t,a,a,g,a,c,a,a,g,g,a,t,t,a,a,a,t,a,t,a,a,a,t,t,t,a,a,t,a,a,a,a,a,a,t,t,t,a,t,c,t,a,g,t,a,a,t,a,t,t,t,t,t,t,t,a,t,a,t,a,t,g,t,t,t,a,t,t,t,a,t,a,a,t,t,t,t,a,a,a,a,a,a,t,a,t,t,c,t,t,t,a,a,a,a,t,t,a,t,t,a,a,a,t,a,t,a,t,t,t,a,a,a,t,t,a,c,c,t,g,a,t,t,g,g,a,a,t,t,t,t,t,t,t,g,a,t,t,g,t,c,c,a,a,a,t,a,t,a,a,a,t,t,a,t,g,a,t,a,a,t,a,t,t,a,t,t,t,a,t,t,a,t,t,c,t,t,c,t,a,t,t,t,t,a,a,a,a,g,a,t,a,a,t,a,a,t,t,t,a,a,t,a,t,a,t,t,a,t,t,t,a,a,a,a,a,a,t,a,a,t,t,t,a,a,a,t,a,t,t,g,a,a,t,t,t,t,t,a,g,a,t,a,g,t,a,t,a,t,t,a,a,t,a,a,a,a,a,a,t,a,a,t,t,c,t,a,t,a,g,a,t,a,t,t,a,t,a,t,t,t,g,a,t,a,g,t,a,t,g,t,c,t,a,t,t,t,t,a,c,a,t,a,c,t,a,c,a,c,a,a,t,a,t,t,t,t,t,t,a,a,a,a,a,a,a,t,t,a,g,g,a,a,t,a,a,t,t,t,t,t,t,t,a,c,c,t,t,t,a,t,t,t,g,a,t,a,t,t,a,t,a,t,t,t,a,a,a,t,a,t,c,c,t,t,t,a,t,t,a,a,t,a,a,a,a,a,a,a,t,a,t,t,t,a,g,g,t,a,c,t,a,t,t,a,t,t,t,c,t,t,a,t,a,a,a,g,a,t,a,a,t,t,t,t,t,t,t,g,c,t,a,a,t,a,t,t,a,a,t,t,c,a,a,t,a,a,t,a,t,t,t,a,g,t,g,a,a,g,g,a,t,c,t,t,t,t,t,g,t,t,a,t,a,t,a,c,c,t,a,a,a,t,a,t,g,t,a,a,a,g,t,g,t,a,a,t,t,t,t,a,a,t,t,t,a,t,c,a,a,c,a,t,a,t,t,t,t,a,a,a,a,c,t,a,a,t,t,c,t,t,c,t,g,a,t,t,t,t,g,c,a,c,a,a,t,t,t,g,a,a,c,g,t,a,c,t,t,t,a,a,t,a,a,t,a,g,t,t,g,g,t,a,a,a,t,a,t,t,c,t,t,a,t,g,t,a,t,c,a,t,a,t,t,t,a,g,a,a,g,g,a,t,g,t,a,c,a,g,c,t,t,c,a,t,t,a,t,a,t,a,a,a,g,a,a,t,c,a,c,a,a,t,t,a,c,a,t,g,t,a,g,c,t,a,t,a,g,t,a,g,a,a,a,t,a,a,t,a,g,t,a,a,a,a,g,a,t,t,a,t,g,g,t,t,a,t,a,t,a,a,a,a,t,a,t,t,a,t,a,c,a,t,t,a,c,a,a,a,a,t,t,g,g,t,a,t,a,g,a,g,g,a,g,a,t,t,a,t,t,t,a,g,g,t,a,a,t,g,g,t,g,g,t,t,t,a,t,a,t,a,a,t,t,t,t,a,c,a,a,c,t,a,a,a,c,g,t,g,g,t,a,t,a,t,g,t,t,t,a,a,a,t,t,a,t,t,c,a,a,a,a,t,t,a,g,a,t,t,g,g,a,t,a,c,a,a,g,t,t,g,a,a,g,t,a,g,g,t,t,c,g,a,t,t,a,t,a,a,c,a,t,g,g,a,a,a,t,a,c,c,c,t,t,c,t,a,c,t,a,t,t,t,t,a,a,a,a,g,g,t,a,a,a,t,t,t,t,c,t,a,t,t,a,g,t,a,a,t,t,t,t,t,a,t,t,c,a,a,t,a,t,c,t,t,t,t,a,t,a,t,c,a,a,a,t,a,t,g,c,a,a,a,t,a,g,c,t,g,a,t,a,c,t,g,g,t,a,g,t,a,a,a,a,t,g,t,a,t,c,a,t,a,t,a,g,g,a,t,c,t,t,a,t,a,c,t,a,a,a,a,g,t,t,a,t,a,t,a,a,t,t,t,c,t,a,a,a,a,g,t,a,t,a,t,c,t,t,t,a,a,a,t,a,a,c,t,c,a,t,t,a,a,a,t,a,t,a,t,t,t,a,g,a,g,g,t,t,t,a,g,t,a,t,a,t,a,t,t,a,a,a,c,c,t,t,t,t,t,c,a,t,a,t,a,a,a,t,c,t,t,a,t,a,a,t,t,a,t,a,c,t,g,a,a,t,g,t,a,g,t,t,c,t,t,t,a,a,t,a,t,t,t,g,g,t,a,a,t,a,a,t,t,c,t,t,t,a,a,c,a,g,t,a,a,c,t,a,t,t,c,c,t,t,a,t,a,t,a,a,a,a,a,a,t,t,a,t,a,a,t,a,a,t,a,c,t,a,g,t,t,a,t,g,t,a,a,a,a,c,a,a,g,a,a,g,c,t,t,t,t,g,t,t,t,c,t,a,a,a,a,t,t,g,a,a,a,t,t,a,t,a,t,a,t,t,t,a,t,t,t,t,t,a,t,t,a,a,t,g,c,a,a,c,g,t,g,g,t,t,t,a,a,g,t,a,t,t,t,c,a,g,a,g,t,c,t,a,t,t,t,c,a,t,t,a,t,t,a,a,t,t,a,t,a,g,g,t,t,t,t,t,g,t,t,c,t,g,a,t,a,t,t,t,a,t,a,a,t,a,a,a,t,t,a,c,c,g,t,t,t,g,a,a,t,t,t,a,a,t,t,t,a,g,a,g,a,t,a,c,c,t,a,t,a,t,t,a,t,t,t,t,c,a,t,t,a,a,a,a,a,t,t,a,a,a,g,a,t,a,t,a,t,t,t,a,a,t,t,a,a],A,B,0).


% p_gener([a,c,a,t,g,c,t], Patgs, Pgt, Pag, Ppar, Pint, Gen, R).

% gen(G), p_gener(G, Patgs, Pgt, Pag, Ppar, Pint, Gen, R).

:- discontiguous p_atgs/4.

:- discontiguous atgs/4.

:- discontiguous buscar/4.

%%%%%%%%%%%% PREDICTORES: ATG, GT, AG, Paradas, Intrones y Gen


%%%%%%%%%%%%  Predictor generalizado: Predictor de sitios ATG, GT, AG, Paradas y estructura para una secuencia problema.

p_gener(Secuencia, Patg, Pgt, Pag, Ppar, Pintrones, Gen, RFSecuencia):-
	p_atgs(Secuencia, Patg, _, 0),
	p_gts(Secuencia, Pgt, _, 0),
	p_ags(Secuencia, Pag, _, 0),
	p_pars(Secuencia, Ppar, _, 0),
	p_gen(Secuencia, Pinicio_atg, Pintrones, Pfinalpar, RFSecuencia),	

	Gen=[Pinicio_atg, Pintrones, Pfinalpar].
%write_triple('PredictGeneral.txt', Pinicio_atg, Pintrones, Pfinalpar)

	



%%%%%%%%%%%%  Predictor ATG: Predictor de sitios de inicio ATG presentes en una secuencia problema.

p_atgs(Secuencia, Posiciones, RSecuencia, Pini):- 
            atgs(Secuencia, Posiciones, Pini, RSecuencia),
	    write_double('atgs.txt', Secuencia, Posiciones).

atgs(Secuencia, Posiciones, PR, RS):-
	senal_atg(Secuencia, 'atg', Patgr, RSecuencia),
	((Patgr==[]-> fail); (PI is (PR + Patgr), NPR is (PI + 3))),
	atgs(RSecuencia, RPosiciones, NPR, RS),
	Posiciones=[PI|RPosiciones], !.

atgs(Secuencia, [], _,Secuencia).

%%%%%%%%%%%%  Predictor GT: Predictor de sitios GT presentes en una secuencia problema.

p_gts(Secuencia, Posiciones, RSecuencia, Pini):- 
            gts(Secuencia, Posiciones, Pini, RSecuencia),
	    write_double('gts.txt', Secuencia, Posiciones).

gts(Secuencia, Posiciones, PR, RS):-
	senal_gt(Secuencia, 'gt', Pgtr, RSecuencia),
	((Pgtr==[]-> fail); (PI is (PR + Pgtr), NPR is (PI + 2))),
	gts(RSecuencia, RPosiciones, NPR, RS),
	Posiciones=[PI|RPosiciones], !.

gts(Secuencia, [], _,Secuencia).


%%%%%%%%%%%%  Predictor AG: Predictor de sitios AG presentes en una secuencia problema.

p_ags(Secuencia, Posiciones, RSecuencia, Pini):- 
            ags(Secuencia, Posiciones, Pini, RSecuencia),
	    write_double('ags.txt', Secuencia, Posiciones).

ags(Secuencia, Posiciones, PR, RS):-
	senal_ag(Secuencia, 'ag', Pagr, RSecuencia),
	((Pagr==[]-> fail); (PI is (PR + Pagr), NPR is (PI + 2))),
	ags(RSecuencia, RPosiciones, NPR, RS),
	Posiciones=[PI|RPosiciones], !.

ags(Secuencia, [], _,Secuencia).

%%%%%%%%%%%%  Predictor Parada: Predictor de sitios de parada presentes en una secuencia problema.

p_pars(Secuencia, Posiciones, RSecuencia, Pini):- 
            pars(Secuencia, Posiciones, Pini, RSecuencia),
	    write_double('paradas.txt', Secuencia, Posiciones).

pars(Secuencia, Posiciones, PR, RS):-
	senal_par(Secuencia, _, Ppar, RSecuencia),
	((Ppar==[]-> fail); (PI is (PR + Ppar), NPR is (PI + 3))),
	pars(RSecuencia, RPosiciones, NPR, RS),
	Posiciones=[PI|RPosiciones], !.

pars(Secuencia, [], _,Secuencia).

%%%%%%%%%%%%  Predictor de intrones GT/AG presentes en una secuencia problema.

p_intrones(Secuencia, Posiciones, RSecuencia, Patg):- 
            intrones(Secuencia, Posiciones, Patg, RSecuencia).

intrones(Secuencia, Posiciones, PR, RS):-
	senal_gt(Secuencia, 'gt', PIr, RSecuencia),
     	senal_ag(RSecuencia, 'ag', PFr, NRSecuencia),
	((PIr==[]; PFr == []) -> fail; (PI is (PR + PIr), PF is (PFr + PI + 2 + 2), NPR is PF)), 
        intrones(NRSecuencia, RPosiciones, NPR, RS),
	Posiciones=[(PI,PF)|RPosiciones], !.

intrones(Secuencia, [], _,Secuencia).

%%%%%%%%%%%%  Predictor de estructura (ATG...GT/AG...TAA/TAG/TGA) para una secuencia problema.

p_gen(Secuencia, Patg, Pintrones, Ppar, RFSecuencia):-
	senal_atg(Secuencia, 'atg', Patg, RSecuencia1),
	(Patg==[]-> PR = 0; PR is Patg + 3),
 	p_intrones(RSecuencia1, Pintrones, RSecuencia2, PR),
	senal_par(RSecuencia2, _, PparR, RFSecuencia),!,
	length(Secuencia,LS), length(RSecuencia2,LRs2),
        (PparR==[]-> Ppar = []; Ppar is (LS - LRs2 + PparR + 3)),
	write_triple('posiciones_gen.txt', Patg, Pintrones, Ppar).


%%%%%%%%%%%
%%%%%%%%%%% Predicados para la exploración de sitios ATG, GT, AG y Parada en secuencias problema

% Reglas senal_atg:

% senal_atg(Secuencia, 'atg', Posicion, RSecuencia) es el predicado a emplear para la busqueda de sitios 
% atg. Se pasa la Secuencia problema, la senal a explorar ATG en este caso. Se debe recibir como resultado la
% Posicion en la que se encuentra un sitio ATG verdadero. 
% Debe devolverse ademas el resto de la secuencia en RSecuencia.

  senal_atg(A, B, Posicion, RSecuencia) :- 
	((conservados_atg(A, B, 'izq', Posicion, RSecuencia)) -> true; 
	((senal(A, B, FailPosicion, _, ZD)-> (senal_atg(ZD, B, NPosicion, RSecuencia), (NPosicion==[] -> Posicion=[];  Posicion is FailPosicion + NPosicion + 3));(fail)))).

  senal_atg(A, _, [], A).
 
% reglas senal_gt:
% En general senal_gt(Secuencia, 'gt', Posicion, RSecuencia) es el predicado a emplear para la busqueda de
% sitios de splicing GT. Se pasa la Secuencia problema, la se~al a explorar (gt, en este caso) 
% se debe recibir como resultado la Posicion en la que se encuentra un 
% sitio gt verdadero, el resto de la secuencia RSecuencia.

  senal_gt(A, B, Posicion, RSecuencia) :- 
	((conservados_gt(A, B, 'izq', Posicion, RSecuencia)) -> true; 
	((senal(A, B, FailPosicion, _, ZD)-> (senal_gt(ZD, B, NPosicion, RSecuencia), (NPosicion==[] -> Posicion=[];  Posicion is FailPosicion + NPosicion + 2));(fail)))).

  senal_gt(A, _, [], A).


% reglas senal_ag, senal_ag(RSecuencia, 'ag', PFr, NRSecuencia)
  senal_ag(A, B, Posicion, RSecuencia) :-
	((conservados_ag(A, B, 'izq', Posicion, RSecuencia)) -> true;
	((senal(A, B, FailPosicion, _, ZD)-> (senal_ag(ZD, B, NPosicion, RSecuencia), (NPosicion==[] -> Posicion=[];  Posicion is FailPosicion + NPosicion + 2));(fail)))).

  senal_ag(A, _, [], A).


% reglas senal_par:
% En general senal_par(Secuencia, 'taa', Posicion, RSecuencia) es el predicado a emplear para la busqueda
% de sitios atg. Se pasa la Secuencia problema, la senal a explorar (TAA, TAG o TGA) en este caso. Se debe
% recibir como resultado la Posicion en la que se encuentra un sitio de parada verdadero. Debe devolverse ademas
% el resto de la secuencia RSecuencia.

  senal_par(A, 'taa', Posicion, RSecuencia) :- 
	((conservados_par(A, 'taa', 'ambas', Posicion, RSecuencia)) -> true; 
% trans_AT(A,_,'taa','no_trans',_), zona_poliadenil(A,'taa','der')
%	((conservados_par(A, 'taa', 'ambas', Posicion, RSecuencia),  trans_GC(A,_,'taa','cod->no_cod',_)) -> true; 
	((senal(A, 'taa', FailPosicion, _, ZD)-> (senal_par(ZD, 'taa', NPosicion, RSecuencia), (NPosicion==[] -> Posicion=[];  Posicion is FailPosicion + NPosicion + 3));(fail)))).

  senal_par(A, 'tag', Posicion, RSecuencia) :- 
	((conservados_par(A, 'tag', 'ambas', Posicion, RSecuencia))-> true; 
% trans_AT(A,_,'tag','no_trans',_), zona_poliadenil(A,'tag','der')
%	((conservados_par(A, 'tag', 'ambas', Posicion, RSecuencia),  trans_GC(A,_,'tag','cod->no_cod',_)) -> true; 
	((senal(A, 'tag', FailPosicion, _, ZD)-> (senal_par(ZD, 'tag', NPosicion, RSecuencia), (NPosicion==[] -> Posicion=[];  Posicion is FailPosicion + NPosicion + 3));(fail)))).

  senal_par(A, 'tga', Posicion, RSecuencia) :- 
	((conservados_par(A, 'tga', 'ambas', Posicion, RSecuencia)) -> true; 
% trans_AT(A,_,'tga','no_trans',_), zona_poliadenil(A,'tga','der')
%	((conservados_par(A, 'tga', 'ambas', Posicion, RSecuencia),  trans_GC(A,_,'tga','cod->no_cod',_) ) -> true; 
	((senal(A, 'tga', FailPosicion, _, ZD)-> (senal_par(ZD, 'tga', NPosicion, RSecuencia), (NPosicion==[] -> Posicion=[];  Posicion is FailPosicion + NPosicion + 3));(fail)))).


  senal_par(A, [], [], A).


%%%%%%%%%%%%%%%% Determina si la Senal ATG satisface la distribucion de conservados 
%%%%%%%%%%%%%%%% En este caso solo hay que insertar lo aprendido en el primer nivel sobre catgi/1, catgd/1.
% => 
conservados_atg(Secuencia, Senal, Direccion, Posicion, Resto):-
	zonas(Secuencia, 8, 8, Senal, Posicion, ZI, ZD, Resto), 
	reverse(ZI, NZI),!,
	d_conservados_atg(NZI,ZD, Direccion).

	d_conservados_atg(NZI, ZD, 'ambas'):- 
			catgi(NZI), catgd(ZD).

	d_conservados_atg(NZI, _, 'izq'):- 
			catgi(NZI).

	d_conservados_atg(_, ZD, 'der'):- 
			catgd(ZD).

	% Estas lineas deben provenir del script gtn.pl

% reglas catgx aprendidas en la primera etapa de aprendizaje.

	catgi(Z) :- b(n,Z,_).
	catgd(Z) :- b(n,Z,_).

	


%%%%%%%%%%%%%%%% Determina si la Senal GT satisface la distribucion de conservados 
%%%%%%%%%%%%%%%% En este caso solo hay que insertar lo aprendido en el primer nivel sobre cgti/1, cgtd/1.
% => 
conservados_gt(Secuencia, Senal, Direccion, Posicion, Resto):-
	zonas(Secuencia, 8, 8, Senal, Posicion, ZI, ZD, Resto), 
	reverse(ZI, NZI),!,
	d_conservados_gt(NZI,ZD, Direccion).

	d_conservados_gt(NZI, ZD, 'ambas'):- 
			cgti(NZI), cgtd(ZD).

	d_conservados_gt(NZI, _, 'izq'):- 
			cgti(NZI).

	d_conservados_gt(_, ZD, 'der'):- 
			cgtd(ZD).

	% Estas lineas deben provenir del script gtn.pl

% reglas cgtX aprendidas en la primera etapa de aprendizaje.

	%cgti(Z) :- b(r,Z,A), b(g,A,B), b(a,B,_).
	%cgtd(Z) :- b(r,Z,A), b(h,A,B), b(v,B,C), b(b,C,D), b(y,D,_).
         

        cgti(Z) :- b(b,Z,A), b(r,A,B), b(a,B,_).


conservados_ag(Secuencia, Senal, Direccion, Posicion, Resto):-
	zonas(Secuencia, 8, 8, Senal, Posicion, ZI, ZD, Resto), 
	reverse(ZI, NZI),!,
	d_conservados_ag(NZI,ZD, Direccion).

	d_conservados_ag(NZI, ZD, 'ambas'):- 
			cagi(NZI), cagd(ZD).

	d_conservados_ag(NZI, _, 'izq'):- 
			cagi(NZI).

	d_conservados_ag(_, ZD, 'der'):- 
			cagd(ZD).

	% Estas lineas debe provenir del script agn.pl

% reglas cagX aprendidas en la primera etapa de aprendizaje.

	%cagi(Z) :- b(g,Z,A), b(a,A,_).

	%cagd(Z) :- b(b,Z,A), b(h,A,B), b(h,B,_),!.
	%cagd(Z) :- b(d,Z,A), b(g,A,_).

        cagi(Z) :- b(b,Z,F), b(r,Z,F), b(r,F,G), b(h,F,G), b(y,G,_).


%%%%%%%%%%%%%%%% Determina si la Senal TAA. TAG o TGA satisface la distribucion de conservados 
%%%%%%%%%%%%%%%% En este caso solo hay que insertar lo aprendido en el primer nivel sobre cpari/1, cpard/1.
% => 
conservados_par(Secuencia, Senal, Direccion, Posicion, Resto):-
	zonas(Secuencia, 8, 8, Senal, Posicion, ZI, ZD, Resto), 
	reverse(ZI, NZI),!,
	d_conservados_par(NZI,ZD, Direccion).

	d_conservados_par(NZI, ZD, 'ambas'):- 
			cpari(NZI), cpard(ZD).

	d_conservados_par(NZI, _, 'izq'):- 
			cpari(NZI).

	d_conservados_par(_, ZD, 'der'):- 
			cpard(ZD).

	% Estas lineas deben provenir del script gtn.pl

% reglas cparX aprendidas en la primera etapa de aprendizaje.

	%cpari(Z) :- b(n,Z,A), b(t,A,B), b(y,B,_), !.
	%cpari(Z) :- b(a,Z,A), b(n,A,B), b(a,B,_).

	%cpard(Z) :- b(y,Z,_).

	cpari(Z) :- b(n,Z,_).
	cpard(Z) :- b(n,Z,_).


% Las versiones de los conceptos difieren ligeramente de aquellas empleadas en las etapas de
% aprendizaje, debido a que en este caso se emplean para predecir y no para entrenar. Notese que lo aprendido
% se pasa como argumentos de los conceptos, tal como sucede con 'no_cod->cod' y 'cod->no_cod' en el concepto
% trans_AT/4.


%%%%%%% Establece si hay alguna regularidad por transicion GC %%%%%%%%
%%% En este caso basta agregar a la regla una nueva variable para capturar la 
%%% posicion. La variable se llama ZD: Zona a la derecha del sitio de splicing.
%%% El valor de ZD sale por RSecuencia al ejecutar la regla trans_AT/5. Lo mismo aplica en el
%%% caso de trans_GC/5.

trans_AT(Secuencia, Posicion, Senal, Tipo_trans, ZD):-
	limite(Limite), 
	zonas(Secuencia, Limite, Limite, Senal, Posicion, ZI, ZD),
	proporcion(ZI, 'at', PI), proporcion(ZD, 'at', PD),
	relacion(PI, PD, R),!,
	tipo_trans(R, Trans),
	Trans==Tipo_trans.

% Busca repeticiones del Dimero en la Secuencia  devuelve la proporcion del Dímero en la
% Secuencia
proporcion(Secuencia, Dimero, Proporcion):-
	length(Secuencia, Longitud),
	Longitud \== 0 -> (buscar(Secuencia, Dimero, Longitud, Repeticiones), Proporcion is float((2*Repeticiones/Longitud))) ; Proporcion is 0.0.

relacion(PI, PD, R):- (PI \== 0.0, PD \== 0.0) -> R is PI/PD.

relacion(PI, PD, R):- (PI < 0.2, PD == 0.0) -> R is 1.0.

relacion(PI, PD, R):- (PI >= 0.2, PD == 0.0) -> R is 1.2.

relacion(PI, PD, R):- (PI == 0.0, PD < 0.2) -> R is 1.0.

relacion(PI, PD, R):- (PI == 0.0, PD >=0.2) -> R is 0.8.

tipo_trans(Relacion, Tipo_trans):- ( (Relacion > 0.8, Relacion < 1.2) -> Tipo_trans = 'no_trans' ). 

tipo_trans(Relacion, Tipo_trans):- (Relacion =< 0.8 -> Tipo_trans = 'no_cod->cod'). 

tipo_trans(Relacion, 'cod->no_cod') :- Relacion >= 1.2.

% Determina la cantidad de polimeros aa en la secuencia.%%%%%%%%%%%%  Predictor ATG: Predictor de sitios de inicio ATG presentes en una secuencia problema.

p_atgs(Secuencia, Posiciones, RSecuencia, Pini):- 
            atgs(Secuencia, Posiciones, Pini, RSecuencia),
	    write_double('atgs.txt', Secuencia, Posiciones).

atgs(Secuencia, Posiciones, PR, RS):-
	senal_atg(Secuencia, 'atg', Patgr, RSecuencia),
	((Patgr==[]-> fail); (PI is (PR + Patgr), NPR is (PI + 3))),
	atgs(RSecuencia, RPosiciones, NPR, RS),
	Posiciones=[PI|RPosiciones], !.

atgs(Secuencia, [], _,Secuencia).
buscar(Secuencia, Polimero, Limite, Cantidad):-   % Dimeros contiene el numero de dimeros gc en en la Secuencia, hasta Limite.
	Polimero == 'at',     			  % En este caso interesa  analizar el dimero 'gt'.
	dat(Posicion, Secuencia, Resto_S),	  % Busca primer dimero aa y devuelve su posicion. Resto_S no incluye al dimero hallado.
	(Nlimite is (Limite - (Posicion + 2))),
	( (Nlimite > 0  -> buscar(Resto_S, Polimero, Nlimite, NCantidad), Cantidad is 1 + NCantidad ) ; Cantidad is 1 ).

% Determina la cantidad de polimeros aa en la secuencia.
buscar(Secuencia, Polimero, Limite, Cantidad):-   % Dimeros contiene el numero de dimeros gc en en la Secuencia, hasta Limite.
	Polimero == 'gc',     			  % En este caso interesa  analizar el dimero 'gt'.
	dgc(Posicion, Secuencia, Resto_S),	  % Busca primer dimero 'aa' y devuelve su posicion. Resto_S no incluye al dimero hallado.
	(Nlimite is (Limite - (Posicion + 2))),
	( (Nlimite > 0  -> buscar(Resto_S, Polimero, Nlimite, NCantidad), Cantidad is 1 + NCantidad ) ; Cantidad is 1 ).

% Halla el primer dimero aa en la secuencia devolviendo su posicion y el resto de la secuencia.

dat(_, [], _) :- true,!, fail.

dat(Posicion, Secuencia, Resto_S) :- 
	at(Secuencia, Resto_S), Posicion is 0.

dat(Posicion, Secuencia, Resto_S) :-
	avanz_pos(Secuencia, Resto),
	at(Resto, Resto_S), Posicion is 1.

dat(Posicion, Secuencia, Resto_S) :-
	avanz_pos(Secuencia, Resto),
	dat(NPosicion, Resto, Resto_S),
	Posicion is (1 + NPosicion).

at(A, B):- 
	b(a, A, C), 
	b(t, C, B).

%%%%%%%%%%%%%%%% Comprueba si hay regularidad por transicion GC %%%%%%%%%%%%%%%%%%%%%%%%%%%
trans_GC(Secuencia, Posicion, Senal, Tipo_trans, ZD):-
	limite(Limite), 
	zonas(Secuencia, Limite, Limite, Senal, Posicion, ZI, ZD), % Resto_Secuencia = ZD,
	proporcion(ZI, 'gc', PI), proporcion(ZD, 'gc', PD),
	relacion(PI, PD, R),!,
	tipo_trans(R, Trans),
	Trans==Tipo_trans.

% Halla el primer dimero aa en la secuencia devolviendo su posicion y el resto de la secuencia.

dgc(_, [], _) :- true,!, fail.

dgc(Posicion, Secuencia, Resto_S) :- 
	gc(Secuencia, Resto_S), Posicion is 0.

dgc(Posicion, Secuencia, Resto_S) :-
	avanz_pos(Secuencia, Resto),
	gc(Resto, Resto_S), Posicion is 1.

dgc(Posicion, Secuencia, Resto_S) :-
	avanz_pos(Secuencia, Resto),
	dgc(NPosicion, Resto, Resto_S),
	Posicion is (1 + NPosicion).

gc(A, B):- 
	b(g, A, C), 
	b(c, C, B).


	
		
%%%%% Determina si la Senal GT precedida por AG NO esta seguida de otra señal GT precedida de AG 

no_gt_vec_prec_ag(Secuencia, Senal, Posicion, RSecuencia):- 
		\+(gt_vec_prec_ag(Secuencia, Posicion, Senal, RSecuencia)).

%%%%% Determina si la Senal GT precedida por AG NO esta seguida de otra señal GT precedida de AG 

no_gt_prec_gt(Secuencia,  Senal, Posicion, RSecuencia):-
		 \+(gt_prec_gt(Secuencia, Senal, Posicion, RSecuencia)).

%%%%% Determina si la Senal GT precedida por AG NO esta seguida de otra señal GT precedida de AG

no_gt_seg_ATn(Secuencia,  Senal, Posicion, RSecuencia):-
		 \+(gt_seg_ATn(Secuencia, Senal,Posicion, RSecuencia)).

%%%%%Determina si la Senal GT esta seguida de un bloque AT mayor a cierto tamaño Umbral 

gt_seg_ATn(Secuencia, Senal, Posicion,  ZD):- 
	(Senal == 'pgt' -> Patron = [g,t]),
	senal(der, Secuencia, Patron, Posicion, ZD),
	umbral_gt_seg_ATn(Umbral), lim_gt_seg_ATn(Limite),
	analisis_bloque_AT(ZD, 'no_inmediato', PosATn, Bloque),
	PosATn < Limite,
	Bloque >= Umbral.

analisis_bloque_AT(Secuencia, Tipo, Posicion, Bloque):- % Devuelve el tamaño del bloque AT y la posicion donde se inicia.
	Tipo == 'no_inmediato', % No importa si el bloque AT es inmediato a la Senal patg.
	dAT(Posicion, Secuencia, Resto_S), % Busca el primer dimero AT y devuelve la posicion en la que se encuentra.
	bloque_AT(Resto_S, NBloque), % Determina el resto del tamaño del Bloque AT.
	Bloque is (NBloque + 2).

analisis_bloque_AT(Secuencia, Tipo, Posicion, Bloque):-
	Tipo == 'inmediato', 
	dAT(Posicion, Secuencia, Resto_S), % Busca el primero dimero AT y devuelve la posicion en la que se encuentra.
	Posicion == 0 -> bloque_AT(Resto_S, Bloque).

bloque_AT(Secuencia, Bloque):- 
	es_AT(Secuencia, Resto)-> bloque_AT(Resto, NBloque), Bloque is (2 +  NBloque).

bloque_AT(_,0).

dAT(_, [], _) :- true,!, fail.

dAT(Posicion, Secuencia, Resto_S) :- 
	es_AT(Secuencia, Resto_S), Posicion is 0.

dAT(Posicion, Secuencia, Resto_S) :-
	avanz_pos(Secuencia, Resto),
	es_AT(Resto, Resto_S), Posicion is 1.

dAT(Posicion, Secuencia, Resto_S) :-
	avanz_pos(Secuencia, Resto),
	dAT(NPosicion, Resto, Resto_S),
	Posicion is (1 + NPosicion).

es_AT(A, B):-
	b(a, A, C), b(t, C, B).

%%%%Determina si la Senal GT precedida por AG esta seguida de otra señal GT seguida de AG %%%%%%%%%
gt_vec_prec_ag(Secuencia, Senal, Posicion, ZD):- 
	(Senal == 'pgt' -> Patron = [g,t]),
	senal_es(Secuencia, Posicion, Patron, 0, Patron, ZI, ZD),!,
	reverse(ZI, NZ), b(g, NZ, R1), b(a, R1, _),
	senal_es(ZD, _, [g,t], 0, [g,t], NZI, _),!,
	reverse(NZI, NNZI),
	length(NNZI, L), dist_max_gt_vec_prec_ag(D),
	L =< D -> b(g, NNZI, R2), b(a, R2, _).

%%% Determina si la Senal GT esta precedida de otra señal GT %%%%%%%%%%%%%%%%%
gt_prec_gt(Secuencia, Senal, Posicion, ZD):- 
	senal(Secuencia, Senal, Posicion, ZI, ZD),!,
	reverse(ZI, Z), 
	b(t, Z, R), b(g, R, _).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Aqui inicia el codigo de aquellos conceptos que pueden componer el cuerpo de las
%% reglas que predicen sitios de splicing AG.  

%%%%%%%%%% Determina si la Senal AG satisface la distribucion de conservados %%%%%%
%% En este caso solo hay que insertar lo aprendido en el primer nivel sobre cagi/1 y cagd/1.
% =>

	
% Determina si la Senal AG presenta una zona rica en pirimidinas hacia la izquierda %%%%%%%%%%%%%
% =>

%reglas pirimidinas

zona_rica_pirimidinas(Secuencia, Senal, Direccion, P, PosSenal, RSecuencia) :- 
             pirimidinas(Secuencia, Senal, Direccion, P, PosSenal, RSecuencia), 
             entre(P, 7), y(P, 12), !. %% Esta linea debe provenir del script agn.pl

zona_rica_pirimidinas(Secuencia, Senal, Direccion, P, PosSenal, RSecuencia) :- 
             pirimidinas(Secuencia, Senal, Direccion, P, PosSenal, RSecuencia), 
             entre(P, 4), y(P, 27). %% Esta linea debe provenir del script agn.pl

pirimidinas(Secuencia, Senal, Direccion, PosCons, PosSenal, ZD):-
	Direccion == izq,
	limite_pirimidinas(Limite), % Establece la longitud de exploracion de bases de ADN 
	senal(Secuencia, Senal, PosSenal, ZI, ZD),
	length(ZI, TZ),
	( ( TZ =< Limite -> NLimite = TZ ) ; NLimite = Limite ),!,
	reverse(ZI, NZI),
	consenso_piri(NZI, NLimite, PosCons, _).

consenso_piri(Secuencia, Limite, Posicion, Resto):-
	yn(Posicion, Secuencia, Resto), V is Limite - 5, Posicion < V, !.	 

% Halla el primer dimero aa en la secuencia devolviendo su posicion y el resto de la secuencia.

yn(_, [], _) :- true,!, fail.

yn(0, Secuencia, Resto):-
      b(y, Secuencia, R1),
      b(y, R1, R2),
      b(y, R2, R3),
      b(y, R3, R4),
      b(y, R4, Resto), !.
      
yn(Posicion, Secuencia, Resto):-
      avanz_pos(Secuencia, R),
      yn(NPosicion, R, Resto),
      Posicion is (1 + NPosicion).


%%%%%%%%%%%%% Determina si la Senal AG presenta zona de ramificacion hacia la izquierda %%%%%%%
% =>

%reglas ramificacion

zona_ramifi(Secuencia, Senal, Direccion, P, PosSenal, RSecuencia) :- 
		ramificacion(Secuencia, Senal, Direccion, P, PosSenal, RSecuencia), 
		entre(P, 35), y(P, 40). %% Esta linea debe provenir del script agn.pl

ramificacion(Secuencia, Senal, Direccion, PosCons, PosSenal, ZD):-
	Direccion == izq,
	limite_ramificacion(Limite), % Establece la longitud maxima de exploracion de bases de ADN a la derecha o izquierda de una señal caracteristica.
	senal(Secuencia, Senal, PosSenal, Zona, ZD), 
	length(Zona, TZ),
	( ( TZ =< Limite -> NLimite = TZ ) ; NLimite = Limite ),!,
	reverse(Zona, NZona),
	consenso_ram(NZona, NLimite, PosCons, _).

consenso_ram(Secuencia, Limite, Posicion, Resto):-
	yyray(Posicion, Secuencia, Resto), V is Limite - 9, Posicion =< V.	 


% Halla el primer dimero aa en la secuencia devolviendo su posicion y el resto de la secuencia.

yyray(_, [], _) :- true,!, fail.

yyray(0, Secuencia, Resto):-
      b(y, Secuencia, R1),
      b(a, R1, R2),
      b(y, R2, R3),
      b(r, R3, R4),
      b(y, R4, R5),
      b(a, R5, R6),
      b(y, R6, R7),
      b(a, R7, R8),
      b(y, R8, Resto), !.

yyray(Posicion, Secuencia, Resto):-
	avanz_pos(Secuencia, R),
	yyray(NPosicion, R, Resto),
	Posicion is (1 + NPosicion).


% Determina si la Senal AG NO esta precedida por otra señal AG%

no_ag_vecino(Secuencia, Senal, Posicion, ZD):- \+(ag_vecino(Secuencia, Senal, Posicion, ZD)).

ag_vecino(Secuencia, Senal, Posicion, ZD):-
	senal(Secuencia, Senal, Posicion, ZI, ZD),!,
	reverse(ZI, Z),
	senal_es(Z, P, [g,a], 0, [g,a], _, _),!,
	dist_max_ag_vecino(D),
	P =< D.

% Determina si la Senal AG NO esta precedida por una zona rica en Aes%

no_ag_polyAes(Secuencia, Senal, Posicion, ZD):- \+(ag_polyAes(Secuencia, Senal, Posicion, ZD)).

ag_polyAes(Secuencia, Senal, Posicion, ZD):-
	senal(Secuencia, Senal, Posicion, ZI, ZD),!,
	reverse(ZI,Z), limite_ag_polyAes(Limite), umbral_ag_polyAes(Umbral),
	( ( Limite =< Posicion -> posicionar(Z, 0, Limite, Zona, _), L = Limite ); ( Zona = Z, L = Posicion ) ),!,
	proporcion_Aes(Zona, L, P),
	P > Umbral.

proporcion_Aes(_, 0, 0.0).
proporcion_Aes(S, L, P):- buscar(S, a, L, Cantidad),!,  P is Cantidad/L.

% Determina si la Senal AG NO esta precedida por varios codones AAG%

no_agpre_aags(Secuencia, Senal, Posicion, ZD):- \+(agpre_aags(Secuencia, Senal, Posicion, ZD)).

agpre_aags(Secuencia, Senal, Posicion, ZD):-
	senal(Secuencia, Senal, Posicion, Z, ZD),!,
	reverse(Z, NZ), limite_agpre_aags(Limite), umbral_agpre_aags(Umbral),
	( (Posicion > Limite -> L = Limite) ; (L = Posicion) ),
	analisis_codones_aag(NZ, L, NCodones), 
	NCodones >= Umbral.

analisis_codones_aag(Secuencia, Limite, NCodones):-
	condon_aag(Posicion, Secuencia, Resto_S), % Busca primer codón aag y devuelve su posicion. 
	NLimite is (Limite - (Posicion + 3)),
	(NLimite > 0 -> analisis_codones_aag(Resto_S, NLimite, NNCod), NCodones is (1 + NNCod)) ; NCodones is 1.

analisis_codones_agg(_, _, 0).

condon_aag(Posicion, Secuencia, Resto_S) :- 
	aag(Secuencia, Resto_S), Posicion is 0.

condon_aag(Posicion, Secuencia, Resto_S) :- 
	avanz_pos(Secuencia, Resto), 
	aag(Resto, Resto_S), Posicion is 1.

condon_aag(Posicion, Secuencia, Resto_S) :-
	avanz_pos(Secuencia, Resto), 
	condon_aag(NPosicion, Resto, Resto_S),
	Posicion is (1 + NPosicion).

aag(A, B):- 
	b(g, A, C),
	b(a, C, D), 
	b(a, D, B).

% Determina si el sitio de parada presenta una zona de poliadenilación aguas abajo %%%%%%%%%%%%%
% =>

zona_poliadenil(Secuencia,Senal,Direccion) :- poliadenil(Secuencia,Senal,Direccion,D), entre(D,29), y(D,70).

poliadenil(Secuencia, Senal, Direccion, PosCons):-
	limite_adenil(Limite), % Establece la longitud maxima de exploracion.
	patron(Senal, Patron),
	senal(Direccion, Secuencia, Patron, _, Zona), 
	length(Zona, TZ),
	( ( TZ =< Limite -> NLimite = TZ ) ; NLimite = Limite ),!,
	consenso_poliadenil(Zona, NLimite, PosCons, _).

consenso_poliadenil(Secuencia, Limite, Posicion, Resto):-
	var(Posicion),
	aataaa(Posicion, Secuencia, Resto), V is Limite - 6, Posicion =< V.	 

consenso_poliadenil(Secuencia, _, Posicion, Resto):-
	not(var(Posicion)),
	posicionar(Secuencia, 0, Posicion, D),
	aataaa(0, D, Resto).

% Halla el consenso AATAAA de la senal de poliadenilacion devolviendo su posicion y el resto de la secuencia.

aataaa(_, [], _) :- true,!, fail.

aataaa(0, Secuencia, Resto):-
      b(a, Secuencia, R1),
      b(a, R1, R2),
      b(t, R2, R3),
      b(a, R3, R4),
      b(a, R4, R5),
      (b(a, R5, Resto);b(n, R5, Resto)).

aataaa(0, Secuencia, Resto):-
      b(a, Secuencia, R1),
      b(a, R1, R2),
      b(a, R2, R3),
      b(a, R3, R4),
      b(a, R4, R5),
      (b(a, R5, Resto);b(n, R5, Resto)).

aataaa(Posicion, Secuencia, Resto):-
      avanz_pos(Secuencia, R),
      aataaa(NPosicion, R, Resto),
      Posicion is (1 + NPosicion).


:-consult(soportes_p_intrones).
:-consult(gen_prueba).
%:-consult(regionTSS).
%:-gen(G), p_gen(G,Patg,Pint,Ppar,RS).
%:-gen(G),p_gener(G,Patgs,Pgt,Pag,Ppar,Pint,Gen,R).

%:-spy(senal_ag/4).
%:-spy(senal_atg/4).
%:-spy(p_intrones/4).
%:-spy(senal_par/4).
%:-spy(senal_gt/4)?
%:-spy(region_poly_A/4).
%:-spy(region_poly_GC/4).
