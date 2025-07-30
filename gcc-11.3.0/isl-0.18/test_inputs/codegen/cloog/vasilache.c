{
  S1();
  S2();
  for (int c0 = 0; c0 < N; c0 += 1)
    for (int c1 = 0; c1 < N; c1 += 1) {
      S4(c0, c1);
      S5(c0, c1);
    }
  for (int c0 = 0; c0 < N; c0 += 1)
    for (int c1 = 0; c1 < N; c1 += 1)
      for (int c2 = 0; c2 <= (N - 1) / 32; c2 += 1) {
        S7(c0, c1, c2, 32 * c2);
        for (int c3 = 32 * c2 + 1; c3 <= min(N - 1, 32 * c2 + 31); c3 += 1) {
          S6(c0, c1, c2, c3 - 1);
          S7(c0, c1, c2, c3);
        }
        if (32 * c2 + 31 >= N) {
          S6(c0, c1, c2, N - 1);
        } else {
          S6(c0, c1, c2, 32 * c2 + 31);
        }
      }
  S8();
}
