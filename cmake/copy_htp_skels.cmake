# Copy libggml-htp-v*.so and, when the build signed them, libggml-htp.cat from
# SRC to DST. A script rather than a copy command because the skel list and the
# catalog's presence are only known after the build ran.
file(GLOB _skels "${SRC}/libggml-htp-v*.so" "${SRC}/libggml-htp.cat")
if(NOT _skels)
    message(WARNING "no HTP skels under ${SRC}")
endif()
foreach(f IN LISTS _skels)
    file(COPY "${f}" DESTINATION "${DST}")
endforeach()
