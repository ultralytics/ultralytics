{
  for (int c0 = 0; c0 <= min(min(T_2 - 1, T_67 - 1), T_66); c0 += 1) {
    S1(c0);
    S2(c0);
  }
  for (int c0 = max(0, T_66 + 1); c0 < min(T_2, T_67); c0 += 1)
    S1(c0);
  for (int c0 = T_67; c0 <= min(T_2 - 1, T_66); c0 += 1) {
    S1(c0);
    S2(c0);
  }
  for (int c0 = max(T_67, T_66 + 1); c0 < T_2; c0 += 1)
    S1(c0);
  for (int c0 = T_2; c0 <= min(T_67 - 1, T_66); c0 += 1)
    S2(c0);
  if (T_2 == 0 && T_67 == 0)
    S1(0);
}
