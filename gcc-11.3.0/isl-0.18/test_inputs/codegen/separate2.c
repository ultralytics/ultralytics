for (int c0 = 0; c0 <= 1; c0 += 1)
  for (int c5 = 0; c5 <= 31; c5 += 1)
    for (int c6 = max(0, 2 * (length % 16) + 2 * c5 - 62); c6 <= 30; c6 += 1) {
      if (2 * length + c6 >= 2 * (length % 16) + 2 && c6 + 62 >= 2 * (length % 16) + 2 * c5 && 2 * (length % 16) >= c6 + 2 && 2 * (length % 16) + 2 * c5 >= c6 && 2 * (length % 32) + c6 == 2 * (length % 16) + 2 * c5 && (2 * c5 - c6) % 32 == 0)
        S_3(c0, 0, (c6 / 2) - (length % 16) + length);
      if (length <= 15 && length >= c5 + 1 && c6 >= 1 && length >= c6)
        S_0(c0, c5, c6 - 1);
    }
