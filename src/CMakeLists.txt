set(FIESTA_LIB_SOURCES
     rk.cpp fiesta.cpp luaReader.cpp bc.cpp reader.cpp
     status.cpp input.cpp output.cpp h5.cpp
     rkfunction.cpp writer.cpp xdmf.cpp block.cpp
)

set(FIESTA_SOURCES
    main.cpp cart2d.cpp
)
     
add_library(FiestaCore STATIC ${FIESTA_LIB_SOURCES})
add_executable(fiesta ${FIESTA_SOURCES})

#Link libraries
target_link_libraries(FiestaCore
    Kokkos::kokkos
    HDF5::HDF5
    ${LUA_LIBRARIES}
    fmt::fmt
    )

target_link_libraries(fiesta FiestaCore)

target_compile_definitions(fiesta     PRIVATE FIESTA_VERSION="1")
target_compile_definitions(FiestaCore PUBLIC  FIESTA_VERSION="1")
target_compile_definitions(fiesta     PRIVATE FIESTA_OPTIONS="${FIESTA_OPTS}")
target_compile_definitions(FiestaCore PUBLIC  FIESTA_OPTIONS="${FIESTA_OPTS}")
target_compile_definitions(fiesta     PRIVATE FIESTA_BTIME="${BUILD_TIME}")
target_compile_definitions(FiestaCore PUBLIC  FIESTA_BTIME="${BUILD_TIME}")

set(DEV_MACRO "HAVE_${DEVICE}")
target_compile_definitions(fiesta     PRIVATE ${DEV_MACRO})
target_compile_definitions(FiestaCore PUBLIC  ${DEV_MACRO})

if(Fiesta_ENABLE_DEBUG)
    target_compile_definitions(fiesta     PRIVATE DEBUG)
    target_compile_definitions(FiestaCore PUBLIC  DEBUG)
endif()

if(Fiesta_SINGLE_PRECISION)
    target_compile_definitions(fiesta     PRIVATE HAVE_SINGLE)
    target_compile_definitions(FiestaCore PUBLIC  HAVE_SINGLE)
endif()

# set install destination
install(TARGETS fiesta RUNTIME DESTINATION)
install(TARGETS FiestaCore RUNTIME DESTINATION)
