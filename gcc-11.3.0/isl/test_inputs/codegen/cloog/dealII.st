domain: "[T_2, T_67, T_66] -> { S1[scat_0] : (scat_0 >= 0 and scat_0 <= -1 + T_2) or (scat_0 <= -T_67 and scat_0 >= 0); S2[scat_0] : (scat_0 >= 0 and scat_0 <= T_66 and scat_0 <= -1 + T_2) or (scat_0 >= 0 and scat_0 <= T_66 and scat_0 <= -1 + T_67) }"
child:
  context: "[T_2, T_67, T_66] -> { [] : T_2 <= 4 and T_2 >= 0 and T_67 <= 4 and T_67 >= 0 }"
  child:
    schedule: "[T_2, T_67, T_66] -> [{ S2[scat_0] -> [(scat_0)]; S1[scat_0] -> [(scat_0)] }]"
    options: "[T_2, T_67, T_66] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[T_2, T_67, T_66] -> { S1[scat_0] }"
      - filter: "[T_2, T_67, T_66] -> { S2[scat_0] }"
